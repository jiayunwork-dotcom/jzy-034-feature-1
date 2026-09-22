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
| `GET  /api/v1/examples/layered-sand` | 细砂/粗砂分层积水入渗算例（毛细屏障） |
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

- `water_content`：逐层给 θ（长度必须等于层数；分层柱按每格所属段的
  [theta_r, theta_s] 校验）；
- `pressure_head`：逐层给压力水头 h；
- `hydrostatic`：给 `water_table_depth_m`，自动建立 h = z − z_wt 的静水平衡剖面
  （分层柱每格用所属段的持水曲线求 θ）。

## 分层土柱（分段土壤剖面）

作业可以在 `material`（整柱一种材料）与 `profile`（分段剖面）之间二选一提交；
两者同时给出会被拒绝（`MATERIAL_PROFILE_CONFLICT`）。`profile` 形如：

```json
{
  "column": {"thickness_m": 1.0, "num_layers": 60},
  "profile": {"layers": [
    {"thickness_m": 0.4, "num_layers": 24,
     "alpha": 7.0, "n": 2.2, "theta_r": 0.05, "theta_s": 0.41, "ks": 6e-5},
    {"thickness_m": 0.6, "num_layers": 36,
     "alpha": 14.0, "n": 2.6, "theta_r": 0.04, "theta_s": 0.43, "ks": 1.2e-4}
  ]},
  "initial":  {"kind": "water_content", "water_content": ["...60 个值..."]},
  "boundary": {"top": "ponded_head", "ponded_head_m": 0.02, "bottom": "free_drainage"},
  "time":     {"total_time_s": 10800, "step_size_s": 60}
}
```

- 段按深度顺序首尾相接，每段自带厚度与一整套 van Genuchten–Mualem 参数；
  段厚度之和必须**精确**等于土柱厚度（否则 `PROFILE_THICKNESS_MISMATCH`），
  每段厚度必须为正，段数至少为 1（一段即退化为单一材料，数值结果与
  `material` 形式逐位一致）。
- 每段的五类参数合法性（n>1、alpha>0、theta_r<theta_s、ks>0、厚度>0）逐段
  校验，错误信息指明出问题的段号（`LAYER_PARAMS_INVALID` 等）。
- 网格按段切分：材料分界面**精确落在计算格点边界上**，没有格点跨段取平均
  参数。段内等分；`num_layers` 可在每段显式给出（总和须等于
  `column.num_layers`），或全部省略由服务按厚度比例分配（大余数法，每段
  至少 1 格）。
- 分界面通量：界面两侧各自用自己的 K(h) 曲线，界面导水率取**距离加权调和
  平均** `Kf = (lu+ld)·Ku·Kd/(ld·Ku + lu·Kd)`（即两个半网格串联阻力；等距时
  退化为普通调和平均），界面两侧共享同一个面通量——通量连续，而压力水头与
  含水量允许跨界面跳变（不同材料在同一 h 下 θ 本就不同）。该加权平均的
  解析导数进入牛顿雅可比，材料导水率相差若干数量级时迭代仍稳定收敛。
- 质量闭合：任意推进区间后整根分层土柱（以及每一段单独）的蓄水变化都精确
  等于进出通量之和，闭合残差保持在 ~1e-10 m 量级，与均质柱相同。
- 响应中的 `material_config` 如实回显作业实际使用的分段结构（各段厚度、
  格点数与参数）；`grid` 额外给出 `face_depths_m`、`segment_of` 与
  `interface_faces`；每步的 `face_fluxes_m_s` 给出全部面通量，可直接核对
  每一段的质量平衡。

单步接口 `POST /api/v1/steps` 同样接受 `profile`，与整段推进共用同一套分段
网格与界面通量/雅可比组装逻辑。

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
可区分、五类非法参数分别被挡、单步/整段同初值同结果、并发多作业互不串扰；
分层柱：两段同参数退化结果与均质逐位一致、非均匀网格质量闭合、分界面通量连续
且两段各自质量平衡、界面两侧各用各的持水/导水曲线、细砂-粗砂界面处锋面明显
滞留（毛细屏障）、强导水率对比下牛顿稳定收敛、按段校验初始剖面与参数、厚度
加总不等于柱厚被拒、单步/整段分层结果一致。
