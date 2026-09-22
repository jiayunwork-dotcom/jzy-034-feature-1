# Richards 非饱和渗流推进服务

一个只做**一件事**的独立 HTTP 服务：按时间推进竖直土柱上的 Richards 方程（van
Genuchten–Mualem 本构、全隐式有限体积、层间串联（调和）导水、全牛顿迭代），逐时刻返回
各深度的含水量 / 压力水头、整柱蓄水量以及上下边界的累计通量。

土柱既可以是**均质**（整柱一套参数），也可以是**分段土壤剖面**：调用方自上而下逐段给出
厚度和该段自己的 alpha、n、theta_r、theta_s、Ks，服务按段切分网格、逐格点使用所属段的
持水曲线与导水率曲线。

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
- 相邻层导水率采用**两点串联（调和）导水**：
  `G = 1/(d_up/K_up + d_down/K_dn)`、`q = (d_up+d_dn)·G + G·(h_up−h_dn)`。
  均质等间距时 `Kf = 2 Ki Kj/(Ki+Kj)`，即经典调和平均（绝不使用算术平均——算术平均在
  锋面跨越数量级时会高估层间通量，使锋面方向/速度失真；测试中有两者可区分的对照用例）。
  **材料分界面**上两侧各用自己的 K(h) 曲线参与串联：分界面只有一个共享面通量（通量连续
  自动成立），两侧压力头与含水量互不约束、通常不相等；该面通量对 h_up、h_dn 的解析导数
  （含各侧自己的 dK/dh）全部进入牛顿雅可比，两侧导水率相差多个数量级时仍稳定收敛。
- 分段网格：材料分界面**精确落在格点面**上，没有任何格点跨越两段材料；段内等间距等分
  （可逐段指定层数，不指定时按厚度用最大余数法分配、总数等于 `num_layers`）。
- 上边界：`ponded_head`（给定表面积水水头，Dirichlet）或 `zero_flux`。
- 下边界（单个算例内钉死其一）：`free_drainage`（单位梯度出流 q=K）或
  `zero_flux`。
- 持水曲线在 h=0 处做了 2 mm 窄带 C1 光滑（Se(0)=1 仍精确成立），仅用于消除
  饱和折点导致的牛顿迭代奇异性；不改变守恒量与上下界。

## 分段土壤剖面

材料通过互斥的两种形式给出：

- 均质（旧形式，继续可用）：顶层 `material` 一个对象；
- 分层：顶层 `materials` 数组，自上而下逐段首尾相接：

```json
"materials": [
  {"thickness_m": 0.4, "num_layers": 20,
   "alpha": 4.5, "n": 2.68, "theta_r": 0.045, "theta_s": 0.43, "ks": 1.7e-5},
  {"thickness_m": 0.6, "num_layers": 30,
   "alpha": 10.0, "n": 2.68, "theta_r": 0.02, "theta_s": 0.36, "ks": 1.2e-4}
]
```

- 段厚度必须为正、段数至少 1，且**厚度之和精确等于** `column.thickness_m`
  （相对容差 1e-10）；不满足直接拒绝并返回 `MATERIAL_INPUT_INVALID`。
- 五类参数（n≤1、alpha≤0、theta_r≥theta_s、Ks≤0）**逐段分别校验**，任一段非法整体
  拒绝，错误字段指明段号，如 `materials[1].ks`。
- 初始含水量逐格点按**所属段**的 `[theta_r, theta_s]` 校验与反算压力头；同一数值在
  砂土段合法、在黏土段可能越界。`hydrostatic` 初值给的是几何压力头分布 h=z−z_wt，
  各格点再用自己那段曲线得到含水量。
- 单段分层（或多段参数完全相同）在数值上与均质路径**逐格点、逐时刻位级一致**。
- 整段推进与单步推进复用同一套分段网格构建、分界面通量与雅可比组装逻辑。

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
  | 分段结构非法（缺段、厚度非正、厚度和≠总厚、段数多于格点、两种材料形式同给） | `MATERIAL_INPUT_INVALID` |

- 作业之间无共享可变状态；每个作业自建 solver 与切片，并发提交互不串扰。

## HTTP 接口

| 方法/路径 | 说明 |
|---|---|
| `POST /api/v1/jobs` | 提交一次完整入渗推进作业 |
| `POST /api/v1/steps` | 只推一个时间步，返回推进前/后剖面与本步闭合残差 |
| `GET  /api/v1/constitutive` | 只读：回显本构公式形式、m–n 绑定、固定常量、材料分界面通量形式 |
| `GET  /api/v1/examples` | 内置算例清单 |
| `GET  /api/v1/examples/sand-ponding` | 均质砂柱积水入渗算例（可直接 POST 回 /jobs） |
| `GET  /api/v1/examples/layered-sand` | 细砂–粗砂分层积水入渗算例（可直接 POST 回 /jobs） |
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

分层时把 `material` 换成 `materials`（二者互斥），其余不变：

```json
{
  "column":   {"thickness_m": 1.0, "num_layers": 50},
  "materials": [
    {"thickness_m": 0.4, "num_layers": 20,
     "alpha": 4.5, "n": 2.68, "theta_r": 0.045, "theta_s": 0.43, "ks": 1.7e-5},
    {"thickness_m": 0.6, "num_layers": 30,
     "alpha": 10.0, "n": 2.68, "theta_r": 0.02, "theta_s": 0.36, "ks": 1.2e-4}
  ],
  "initial":  {"kind": "water_content",
               "water_content": ["20 个细砂段内的值", "30 个粗砂段内的值"]},
  "boundary": {"top": "ponded_head", "ponded_head_m": 0.02,
               "bottom": "free_drainage"},
  "time":     {"total_time_s": 14400, "step_size_s": 120}
}
```

`materials[].num_layers` 可省略（按厚度比例分配，总数取 `column.num_layers`），
也可逐段显式给出，但显式给出时之和必须等于 `column.num_layers`。

`initial.kind` 支持：

- `water_content`：逐层给 θ（长度必须等于层数）；
- `pressure_head`：逐层给压力水头 h；
- `hydrostatic`：给 `water_table_depth_m`，自动建立 h = z − z_wt 的静水平衡剖面。

`step_size_s` 是**输出/报告**时间间隔；当某个区间内牛顿迭代困难时服务会自动在
内部二分（不改变输出时刻），并在每个区间返回实际使用的 `substeps` 数。单步接口
严格使用调用方给定的步长，不做内部切分，不收敛即报错。

响应含逐时刻 `layers[].{depth_m, water_content, pressure_head_m, segment}`、每步
`storage_m`、`cum_top_flux_m`、`cum_bottom_flux_m`、`mass_balance_residual_m`、
`iterations`、`substeps`，以及整段的 `total_mass_balance_residual_m`。分层作业还回显
`materials[]`（各段厚度、五类参数与所属格点区间）和 `grid` 中的
`cell_thickness_m_all`、`segment_of_cell`、`segment_interface_depths_m`。

## 内置算例

- 均质砂柱（alpha=6 /m，n=2，thetaR=0.05，thetaS=0.40，Ks=5e-5 m/s），初始均匀
  θ=0.15，顶面 2 cm 积水，底面自由排水，推进 90 分钟。湿润锋随时间单调下移，蓄水
  持续增加，顶边界累计入流与蓄水增量一致（闭合残差 ~1e-13 m）。
- 分层细砂–粗砂（0.4 m 细砂 + 0.6 m 粗砂，分界面精确落在格点面上）。粗砂 alpha
  更大，锋前非饱和状态下其导水率远低于细砂（尽管 Ks 高约 7 倍），形成毛细屏障：
  锋面到达分界面后明显滞留，待界面前水头抬升到粗砂进气条件后才突破，突破后推进
  速度也显著改变；分界面两侧含水量各自落在本段区间内、互不相等，而两侧反算的
  面通量是同一个连续值，整柱质量闭合仍维持 ~1e-12 m。

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
以及分层专项：两段参数相同（及单段分层）与均质路径逐格点逐时刻位级一致、
细砂–粗砂锋面在分界面前滞留、突破前后速度可观测变化（几何/初值匹配的均质对照
无此变化）、分界面共享通量连续且两侧格点残差为零、分界面雅可比与有限差分吻合、
分界面两侧含水量各按本段曲线落在本段区间、分层质量闭合维持原有精度量级、
初始含水量校验按格点所属段分别生效、分段参数逐段校验且错误指明段号、
段厚度和≠总厚（及段数多于格点、两种材料形式同给）被拒绝、分层下单步与整段
同初值同结果、分层零通量静水柱保持不动。
