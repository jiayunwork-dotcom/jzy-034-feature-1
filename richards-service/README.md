# Richards 非饱和渗流推进服务

一个只做**一件事**的独立 HTTP 服务：按时间推进竖直土柱上的 Richards 方程（van
Genuchten–Mualem 本构、全隐式有限体积、层间调和平均、全牛顿迭代），逐时刻返回
各深度的含水量 / 压力水头、整柱蓄水量以及上下边界的累计通量。

- 语言/框架：Go 1.22 + Gin
- 单位：长度 m、时间 s、导水率 m/s、通量 m/s（向下为正），深度 z 向下为正

## 目录结构（按职责分包）

```
cmd/server/                 服务入口（仅装配与启动）
internal/constitutive/      van Genuchten 持水曲线 + Mualem 相对导水率（含解析导数）
internal/solver/            网格 / 层间调和平均 / 有限体积残差 / 全隐牛顿推进
                            grid.go      差分格子与层间导水率（调和、算术）
                            solver.go    单步全隐牛顿求解（Step）
                            adaptive.go  报告区间内的自适应子步切分（StepAdaptive/MarchAdaptive）
internal/job/               入参校验、作业编排、作业状态/计数、内置砂柱算例
internal/api/               Gin 路由与结构化错误
```

单步推进与整段推进复用**同一套本构函数、同一个差分格子、同一个 `Step`**：整段
推进只是反复调用 `Step`（外层 `StepAdaptive` 仅在某个报告区间牛顿不收敛时把区间
内部二分，边界累计通量与蓄水量仍逐子步严格记账）。

## 本构与离散（固定形式）

- 有效饱和度（h<0 时）：
  `Se = [1 + (alpha*|h|)^n]^(-m)`，**m 与 n 锁死为 m = 1 − 1/n**；h ≥ 0 时 Se = 1。
- 含水量：`theta = theta_r + (theta_s − theta_r) * Se`。
- Mualem 相对导水率（缩减因子 l = 1/2 固定）：
  `Kr = Se^(1/2) * [1 − (1 − Se^(1/m))^m]^2`，`K = Ks * Kr`。
- Richards 方程（z 向下）：`dtheta/dt = d/dz [K(h) * (dh/dz − 1)]`。
- 时间：全隐向后欧拉；每步全牛顿迭代（雅可比含面导水率解析导数 + Armijo 回退
  线搜索），残差与水头增量同时收敛才认步。
- 相邻层导水率：**调和平均** `Kf = 2 Ki Kj/(Ki+Kj)`，绝不使用算术平均（算术平均
  在锋面跨越数量级时会高估层间通量，使锋面方向/速度失真；测试中有两者可区分
  的对照用例）。
- 上边界：`ponded_head`（给定表面积水水头，Dirichlet）或 `zero_flux`。
- 下边界（单个算例内钉死其一）：`free_drainage`（单位梯度出流 q=K）或
  `zero_flux`。
- 持水曲线在 h=0 处做了 2 mm 窄带 C1 光滑（Se(0)=1 仍精确成立），仅用于消除
  饱和折点导致的牛顿迭代奇异性；不改变守恒量与上下界。

## 守恒与失败语义（硬保证）

- 每个接受步（及每个报告区间）都返回质量闭合残差：
  `ΔStorage − (qTop − qBot)·Δt`，在数值精度（约 1e-10 m 水柱）内为 0。
- 所有时刻每个格点含水量严格落在 `[theta_r, theta_s]`；越界即作业失败，
  返回 `THETA_OUT_OF_RANGE`，**绝不削平越界值**。
- 牛顿不收敛返回 `NON_CONVERGENCE`，失败步被整体拒绝、求解器状态不前进。
- 五类非法本构/几何参数在推进前拦截，返回带类型的结构化错误：

  | 条件 | code |
  |---|---|
  | n ≤ 1 | `N_NOT_GREATER_THAN_ONE` |
  | alpha ≤ 0 | `ALPHA_NON_POSITIVE` |
  | theta_r ≥ theta_s | `THETA_R_GE_THETA_S` |
  | Ks ≤ 0 | `KS_NON_POSITIVE` |
  | 土柱厚度 ≤ 0 | `COLUMN_THICKNESS_NON_POSITIVE` |

- 作业之间无共享可变状态；每个作业自建 solver 与切片，并发提交互不串扰。

## HTTP 接口

| 方法/路径 | 说明 |
|---|---|
| `POST /api/v1/jobs` | 提交一次完整入渗推进作业 |
| `POST /api/v1/steps` | 只推一个时间步，返回推进前/后剖面与本步闭合残差 |
| `GET  /api/v1/constitutive` | 只读：回显本构公式形式、m–n 绑定、固定常量 |
| `GET  /api/v1/examples` | 内置算例清单 |
| `GET  /api/v1/examples/sand-ponding` | 砂柱积水入渗算例（可直接 POST 回 /jobs） |
| `GET  /healthz` | 存活探针 |
| `GET  /status` | 基本运行状态/监控计数 |

### 作业请求示例

```json
{
  "column":   {"thickness_m": 1.0, "num_layers": 50},
  "material": {"alpha": 6.0, "n": 2.0, "theta_r": 0.05, "theta_s": 0.40, "ks": 5e-5},
  "initial":  {"kind": "water_content",
               "water_content": [0.15, 0.15, "..."]},
  "boundary": {"top": "ponded_head", "ponded_head_m": 0.02,
               "bottom": "free_drainage"},
  "time":     {"total_time_s": 5400, "step_size_s": 30}
}
```

`initial.kind` 支持：

- `water_content`：逐层给 θ（长度必须等于层数）；
- `pressure_head`：逐层给压力水头 h；
- `hydrostatic`：给 `water_table_depth_m`，自动建立 h = z − z_wt 的静水平衡剖面。

`step_size_s` 是**输出/报告**时间间隔；当某个区间内牛顿迭代困难时服务会自动在
内部二分（不改变输出时刻），并在每个区间返回实际使用的 `substeps` 数。单步接口
严格使用调用方给定的步长，不做内部切分，不收敛即报错。

响应含逐时刻 `layers[].{depth_m, water_content, pressure_head_m}`、每步
`storage_m`、`cum_top_flux_m`、`cum_bottom_flux_m`、`mass_balance_residual_m`、
`iterations`、`substeps`，以及整段的 `total_mass_balance_residual_m`。

## 内置砂柱算例

1 m 砂柱（alpha=6 /m，n=2，thetaR=0.05，thetaS=0.40，Ks=5e-5 m/s），初始均匀
θ=0.15，顶面 2 cm 积水，底面自由排水，推进 90 分钟。湿润锋随时间单调下移
（约 0.15 → 0.35 → 0.57 → 0.79 m），总蓄水量持续增加，顶边界累计入流与蓄水
增量一致（闭合残差 ~1e-13 m）。

## 运行

```bash
# 本地
go run ./cmd/server                 # 默认 :8080，可用 LISTEN_ADDR 覆盖

# Docker（golang:1.22-alpine 多阶段构建）
docker build -t richards-service .
docker run --rm -p 8080:8080 richards-service

# 试一下
curl -s localhost:8080/api/v1/examples/sand-ponding | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["request"]))' \
  > /tmp/req.json
curl -s -X POST localhost:8080/api/v1/jobs -H 'Content-Type: application/json' \
  --data @/tmp/req.json
```

## 测试

```bash
go test ./...              # 全部行为测试
go test -race ./...        # 含并发竞态检测
```

覆盖：单步与整段的质量闭合、增大积水水头锋面更深且蓄水更多、降一个数量级 Ks
锋面明显变慢、零通量静水剖面长时间不动、含水量全程不出界、调和与算术平均结果
可区分、五类非法参数分别被挡、单步/整段同初值同结果、并发多作业互不串扰。
