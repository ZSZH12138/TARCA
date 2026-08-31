# Stage 2 概率预测器、E02 正式验收与双 RTX 4090 服务器运行设计

> 状态：`USER_APPROVED_SCIENTIFIC_RULES_PENDING_WRITTEN_SPEC_REVIEW`
> 日期：2026-08-31
> 方案：`B`
> Stage 2 科学身份：`stage2_probabilistic_forecasting_v1`
> E02 科学身份：`e02_predictor_validity_v1`
> 协议身份：`TARCA-E2E-STAGE-PROTOCOL-2.0`
> 当前边界：本文件冻结实施设计；尚未连接、上传或启动服务器，尚未访问 E02 正式分区

## 1. 目的与范围

Stage 2 把 Stage1B 已确认的双尺度 Lorenz-96 世界转换为可供后续机制实验使用的概率预测器。
它要完成数据桥接、基线、DLinear、PatchTST、iTransformer、概率输出、校准、训练、冻结和
可复现打包。E02 随后使用事先封存的正式数据，对 Stage 2 已冻结的预测器做一次预测有效性
验收。

本设计同时把本地实施、服务器预检、双 GPU/CPU 调度、故障恢复、只读监控和最终判定写成
一个不可随运行结果修改的合同。实施完成后，应能把固定 bundle 放到指定服务器，先完成
preflight，再在用户明确授权后直接启动，无需临时改代码、补配置或重新决定科学规则。

本设计包含：

1. Stage 2 与 E02 的科学规则和身份；
2. 数据、seed、split、模型、概率分布、指标和判定；
3. 本地代码、配置、容器、离线来源 capsule 和测试；
4. 两张 RTX 4090、28 CPU、224GB RAM 的有用饱和调度；
5. 服务器授权边界、24 小时环境重置、恢复和审计。

本设计不包含：

- 不连接服务器、不上传文件、不启动容器；
- 不运行完整 Stage 2 训练或 E02；
- 不读取、生成或评分 E02 正式结果；
- 不修改 Stage1B 或 E01 的既有科学结论；
- 不进入 Stage 3/4，不执行内部激活干预；
- 不因 ETA、显存或中间结果减少 seed、模型、epoch、trajectory 或 horizon；
- 不把 GPU 百分比本身当作目标，不运行无科学用途的占位计算。

## 2. 权威文件与上游身份

本设计服从 `docs/auth` 中的计划书、具体实施计划、协议规范、契约书、变更控制文件、Stage1B
交接快照、E01 交接快照和服务器接入 runbook。若实现细节与权威文件冲突，先 fail closed，
不得用代码覆盖协议含义。

固定上游输入如下：

| 项目 | 固定值 |
|---|---|
| 世界 | `lorenz96_twoscale_v2` |
| Stage1B 活动 manifest SHA-256 | `d1b4d09260bcc41b3b94a020474ee0b5e9f9dd5f0f498bb96510228141f44b25` |
| E01-v2 收据内部 SHA-256 | `16de7fc103b8f1589eec07deaebfb66fbf7ea603046020e4778bb52458c3ae14` |
| history | `64` |
| forecast horizon | `24` |
| trajectory length / warmup | `512 / 0`，沿用已冻结 Stage1B v2 |
| 主时距 | `h1-6` |
| 次时距 | `h7-12`、`h13-24` |
| 正式数据 seeds | `1729, 2718, 3141, 5772, 8111` |
| 服务器基础镜像 | `pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04` |
| 目标硬件 | `2 × RTX 4090 24GB, 28 CPU, 224GB RAM` |

Stage1B 证明该世界和主时距足以支持神经预测器相对 VAR 的独立确认，但它不是 Stage 2 的
正式训练结果，也不能代替 E02。E01 证明 paired effect oracle 可作为后续效果测量尺，但它
不评价预测器 NLL/CRPS，也不能代替 E02。Stage 2 只消费标准 `WindowBatch`/split bridge；
普通预测器不能读取 SCM truth、生成器私有状态或 E01 paired effect truth。

## 3. 论文、官方资料与源码来源

所有外部源码必须固定 URL、commit、取得日期、许可证状态和内容哈希，并进入离线 source
capsule。服务器正式运行不得临时从 GitHub 拉取“最新版本”。

| 来源 | 固定用途 | 固定身份或方法依据 |
|---|---|---|
| [DLinear 官方仓库](https://github.com/cure-lab/LTSF-Linear) | DLinear mean model | commit `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`，kernel `25` |
| [PatchTST 官方仓库](https://github.com/yuqinie98/PatchTST) | PatchTST backbone | commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9` |
| [Time-Series-Library](https://github.com/thuml/Time-Series-Library) | iTransformer backbone | commit `4e938a1767106324dd753b2a44832bf870a0252e` |
| [Lorenz-96 probabilistic forecasting source](https://github.com/Climdyn/lorenz96-forecasts) | 双尺度世界复核 | commit `6f28942f6a703c2b52501d01258ca2708539f209` |
| [Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437) | proper scoring rule、CRPS/NLL | 预测分布以 proper score 评价 |
| [Gneiting et al., 2007](https://doi.org/10.1111/j.1467-9868.2007.00587.x) | calibration 与 sharpness | 覆盖率护栏不能由单一 point metric 代替 |
| [Diebold & Mariano, 1995](https://doi.org/10.1080/07350015.1995.10524599) | 配对预测误差比较 | 同一 trajectory 上做配对差值 |
| [Gilleland, 2020](https://doi.org/10.1175/JTECH-D-20-0069.1) | 时序 block bootstrap | 重采样单位保留时序相关性 |
| [Bouthillier et al., 2019](https://proceedings.mlr.press/v97/bouthillier19a.html) | 多随机源复现 | 数据 seed 与初始化 seed 分离报告 |
| [DLinear 论文](https://doi.org/10.1609/aaai.v37i9.26317) | 线性强基线 | 不能只与 naive 比较 |
| [PatchTST 论文](https://openreview.net/forum?id=Jbdc0vTOcol) | patch-based neural baseline | 与官方主干保持一致 |
| [iTransformer 论文](https://openreview.net/forum?id=JePfAI8fah) | 变量 token 主模型 | 与 Stage1B 选定路线衔接 |

来源物化失败、commit 漂移、哈希漂移或官方 wrapper 等价测试失败都阻断正式启动。许可证
状态写入 provenance，但本文件不作新的法律结论。

## 4. 科学身份与不可变量

Stage 2 和 E02 各自生成独立 manifest。manifest 至少包含：上游哈希、数据 manifest、split、
normalizer、模型和来源 commit、超参数、训练 seed、precision、指标、bootstrap、阈值、硬件
无关的科学 task identity 以及授权事件。

以下内容属于 Science Plane，正式 launch 后不可修改：

- 世界、history、horizon、变量顺序和窗口规则；
- trajectory、partition、data seed 和 model-init seed；
- normalizer 拟合范围；
- 模型集合、超参数、训练预算、early stopping 和 checkpoint 选择；
- 输出分布族、scale 下界和 residual calibration；
- 主比较、次比较、指标、bootstrap 和门槛；
- PASS/FAIL/INCONCLUSIVE 规则；
- precision policy。

以下内容属于 Execution Plane，可以根据预注册遥测规则调整，但不得改变 scientific identity：

- task placement、CPU affinity、DataLoader worker 数；
- 每卡同时处理的推理 bundle 数；
- 独立任务在 GPU0/GPU1 的顺序；
- 断点恢复、失败后降低同卡并发和一次重试；
- DDP 是否启用，但只能使用本设计规定的代表性探针和 30% 收益门槛。

## 5. 数据、seed 与 split

### 5.1 开发数据 seed

Stage 2 新增三个开发数据 seed，与 Stage1B 资格、E01 和 E02 正式 seed 完全隔离：

| namespace | seed |
|---|---:|
| `tarca/stage2_probabilistic_forecasting_v1/dev-data/0` | `669591429` |
| `tarca/stage2_probabilistic_forecasting_v1/dev-data/1` | `1840764098` |
| `tarca/stage2_probabilistic_forecasting_v1/dev-data/2` | `1185077341` |

seed 生成算法固定为：对 UTF-8 namespace 计算 SHA-256，把前 8 bytes 按 little-endian 无符号
整数解析，再取 `1 + value mod 2147483646`。编译器必须再次检查它们不出现在所有上游
qualification/E01/formal seed 清单中；发现碰撞即报错，不自动换 seed。

每条 trajectory 长度固定为 `512`，warmup 固定为 `0`。每个开发 seed 生成：

- `24` 条 `TRAIN` trajectory；
- `8` 条 `VALIDATION` trajectory。

合计 `72 TRAIN + 24 VALIDATION = 96` 条完整 trajectory。所有轨迹先生成 manifest 和内容
哈希，再构造重叠窗口。窗口是训练样本，但不是统计独立单元。

### 5.2 正式 E02 数据

E02 固定使用五个正式数据 seed：`1729, 2718, 3141, 5772, 8111`。每个 seed 包含：

- `12 TEST_SEEN_REGIME` trajectory；
- `12 TEST_UNSEEN_REGIME` trajectory。

合计 `60 seen + 60 unseen = 120` 条完整 trajectory。每个 formal seed × regime 形成一个
bootstrap stratum，每个 stratum 恰有 12 个 trajectory。

Stage 2 的普通 prepare、dry-run、单元测试、服务器性能探针和训练任务都不得物化、打开或
统计这 120 条正式 trajectory。只有 E02 formal grant 校验成功后，formal worker 才能从
sealed manifest 解封路径。路径访问和 artifact read 都写入审计日志。

### 5.3 normalizer 与窗口

- normalizer 只在 `TRAIN` trajectory 上拟合；
- `VALIDATION`、seen 和 unseen 只使用已冻结 normalizer；
- normalizer 必须记录变量顺序、统计量、训练 trajectory hashes 和自身 SHA-256；
- 每个输入窗口 history=`64`，输出 horizon=`24`；
- 不跨 trajectory、regime 或 partition 边界构造窗口；
- 不用未来窗口、formal 数据或 truth sidecar 拟合任何数据变换；
- CRPS/NLL/MAE 的最终独立重采样单位是完整 trajectory，不是重叠窗口。

## 6. 模型集合与概率输出

### 6.1 固定模型

Stage 2 必须构建并冻结以下六类 predictor：

1. `LastValueGaussian`：最后观测值作为所有 horizon 的均值；
2. `SeasonalNaiveGaussian`：lag 从 `[8, 16, 32]` 中只按 validation CRPS 选择；
3. `VARGaussian`：复用现有 VAR，lags=`[1,2,4,8,16]`、ridge=`[1e-6,1e-3,1e-1]`；
4. `DLinearGaussian`：官方 DLinear，moving-average kernel=`25`；
5. `PatchTSTGaussian`：官方 PatchTST 薄适配；
6. `ITransformerGaussian`：官方 iTransformer 薄适配，也是 E02 主神经模型。

Last-value 与 Seasonal-naive 是弱基线和 sanity guardrail。VAR 与 DLinear 是线性候选，二者
只按 Stage 2 validation h1-6 CRPS 选择“最强线性基线”。PatchTST 是次神经比较；
iTransformer 是预注册主神经模型，不能根据 E02 结果改成 PatchTST。

### 6.2 线性/naive 概率尺度

Last-value、Seasonal-naive 和 DLinear 的 mean 与概率尺度分开拟合。尺度只能使用 TRAIN：

1. 以完整 TRAIN trajectory 为分组做固定五折 cross-fitting；
2. fold 分配由 `SHA256(trajectory_id) mod 5` 决定，不打乱窗口；
3. 对每折只在另外四折拟合 mean model，对留出折产生 out-of-fold residual；
4. 对每个 `horizon × variable` 计算 `sqrt(mean(residual^2))`；
5. 在标准化空间把 scale 下界固定为 `1e-4`，上界固定为
   `max(10.0, 10 × TRAIN target standard deviation)`；
6. 用全部 TRAIN 重新拟合最终 mean model，但保留 cross-fitted scale；
7. de-normalize 时同时按训练标准差变换 mean 和 scale。

Seasonal lag 的选择只使用 VALIDATION；它的每个候选 lag scale 仍只由 TRAIN cross-fitting
获得。VAR 使用 TRAIN residual covariance 和递推 forecast covariance，输出合同只持久化其
正对角 scale；协方差加 `1e-6 I` 后仍非正定或产生非有限值则模型无效。

DLinear 固定 `individual=false`，mean loss 为 MSE，batch=`64`，max epochs=`100`，
patience=`20`，AdamW learning rate=`1e-3`、betas=`(0.9, 0.999)`、eps=`1e-8`、
weight decay=`0`，gradient clip norm=`1.0`。五个 cross-fit fold 的初始化 seed 按 fold 0–4
依次固定为 `374158318, 2032114884, 152378261, 341781751, 1071263255`；它们由
`tarca/stage2_probabilistic_forecasting_v1/dlinear-fold/<fold>` 按 5.1 的算法生成。最终全量
TRAIN fit 使用 seed `1797287582`。cross-fit 只产生概率尺度，最终 DLinear mean checkpoint
在 VALIDATION MSE 上 early stop，随后和 VAR 一起按 VALIDATION h1-6 CRPS 选择最强线性基线。

### 6.3 神经概率头

PatchTST 与 iTransformer 直接输出每个 horizon × variable 的 `mean` 和 `raw_scale`：

```text
scale = softplus(raw_scale) + 1e-4
distribution = Independent(Normal(mean, scale), variable_dimension)
```

训练目标是 diagonal Gaussian NLL。validation NLL 用于 early stopping，validation CRPS 用于
预注册的最终 checkpoint/model-init 选择。概率头不得在 formal 数据上 temperature scaling、
variance rescaling 或再次校准。

### 6.4 神经超参数

PatchTST 固定沿用 Stage1B：`d_model=128`、`layers=3`、`heads=16`、`d_ff=256`、
`dropout=0.1`、`patch_len=16`、`stride=8`、RevIN、batch=`64`、max epochs=`100`、
patience=`20`、learning rate=`1e-4`。

iTransformer 固定沿用 Stage1B：`d_model=512`、`layers=3`、`heads=8`、`d_ff=512`、
`dropout=0.1`、RevIN、batch=`32`、max epochs=`100`、patience=`20`、learning rate=`1e-4`。

优化器、weight decay、gradient clipping、AMP dtype 和 deterministic setting 必须写入配置和
manifest。两种神经模型统一使用 AdamW：betas=`(0.9, 0.999)`、eps=`1e-8`、weight
decay=`0.01`、gradient clip norm=`1.0`，不使用 learning-rate scheduler。固定启用
`torch.use_deterministic_algorithms(True)` 与 cuDNN deterministic，关闭 cuDNN benchmark；
DataLoader shuffle 由对应 model-init seed 驱动。服务器 preflight 可以在事先声明的 FP32 与
AMP-FP16 两个候选中比较；AMP dtype 固定为 `float16`。只有数值
一致性通过后才能选择更快者，并在 formal launch 前冻结。不得在正式运行中自适应切换。

### 6.5 模型初始化 seed

两个神经模型各自训练三个固定 initialization：

| namespace | seed |
|---|---:|
| `tarca/stage2_probabilistic_forecasting_v1/model-init/0` | `1797287582` |
| `tarca/stage2_probabilistic_forecasting_v1/model-init/1` | `883082243` |
| `tarca/stage2_probabilistic_forecasting_v1/model-init/2` | `1933050005` |

每个初始化都保存 best-NLL checkpoint、validation 预测和最终收据。每个架构只按 validation
h1-6 CRPS 选一个 primary checkpoint；其余两个不删除，用于 initialization stability 护栏。
禁止第四次 reroll，禁止在看到 E02 后改选初始化。

## 7. Stage 2 冻结输出

Stage 2 只有在开发数据完全有效、所有六类 predictor 可重载、概率分布有效、validation 选择
完成且来源/环境可复现后，才能生成 freeze candidate。固定输出包括：

- Stage 2 manifest 与完整配置哈希；
- data/split/normalizer manifests；
- 六类 predictor 的 model identity；
- 两个神经模型各三个 checkpoint 及权重哈希；
- validation-only model selection receipt；
- 最强线性基线身份；
- iTransformer primary initialization 身份；
- source capsule、环境 lock 和容器 identity；
- 运行账本、失败清单、precision record 和资源探针收据；
- E02 input bundle，只包含冻结身份，不包含 formal 结果。

freeze 命令拒绝缺失 artifact、hash drift、非有限预测、非正 scale、formal access event、
未声明重试或 validation 选择不一致。Stage 2 freeze 不代表 E02 PASS。

## 8. E02 主问题、指标和门槛

### 8.1 唯一主比较

E02 的唯一主比较是：

```text
冻结的 iTransformer primary initialization
vs
Stage 2 validation 预先选出的最强线性基线（VAR 或 DLinear）
```

主指标只使用 h1-6 的 CRPS：

```text
CRPS skill = 1 - mean_CRPS(iTransformer) / mean_CRPS(linear_baseline)
```

mean 先在每条 trajectory 内对 origin、horizon 1–6 和 variable 做等权平均，再对 120 条
trajectory 等权平均。seen/unseen 和 seed 分层报告使用同样方法。

### 8.2 配对分层 block bootstrap

bootstrap seed 固定为 `172657089`，namespace 为
`tarca/stage2_probabilistic_forecasting_v1/bootstrap/0`。执行 `5000` 次：

1. 保持同一 trajectory 上神经与线性预测的配对关系；
2. 在 5 formal seeds × 2 regimes 的 10 个 strata 内分别有放回抽取 12 条完整 trajectory；
3. 每次合并 120 条抽样 trajectory 计算 CRPS skill；
4. 使用 percentile 5th 与 95th percentiles 构造 90% 两侧区间；
5. 区间下界必须严格大于 0；这等价于预注册方向上的 one-sided 95% 下界大于 0。

禁止把重叠窗口当作独立 bootstrap 单元，禁止挑选有利 seed 或 regime。

### 8.3 PASS 的全部条件

E02 只有同时满足以下条件才是 `PASS`：

1. 主 h1-6 CRPS skill `>= 0.02`；
2. 90% bootstrap CI 下界 `> 0`；
3. 五个 formal data seed 中至少 `3/5` 的 h1-6 skill `> 0`；
4. 三个 iTransformer initialization 中至少 `2/3` 对同一线性基线的 h1-6 skill `> 0`；
5. primary iTransformer 的总体 CRPS 同时优于 Last-value 和 Seasonal-naive；
6. seen h1-6 skill `> 0`；
7. unseen h1-6 skill `>= -0.05`；
8. aggregate NLL 相对最强线性基线的恶化不超过 `5%`；
9. aggregate MAE 相对最强线性基线的恶化不超过 `5%`；
10. 50%、80%、90%、95% central interval 的平均绝对 coverage error 总体 `<= 0.05`；
11. seen 和 unseen 各自的平均绝对 coverage error `<= 0.10`；
12. h7-12 CRPS skill `>= -0.10`；
13. h13-24 CRPS skill `>= -0.10`；
14. 预测、NLL、CRPS、MAE 和区间全部有限，所有 scale 严格大于 0；
15. 所有输出分位数单调，无 quantile crossing；
16. 120/120 正式 trajectory 完成，科学输入和输出 hash 完整，未发生选择性重跑。

NLL/MAE 的“恶化不超过 5%”精确定义为：当 baseline metric 为正时，
`neural_metric / baseline_metric <= 1.05`。coverage error 定义为四个 nominal levels 上
`abs(empirical_coverage - nominal_coverage)` 的算术平均。

### 8.4 PASS、FAIL、INCONCLUSIVE 与 NOT_EVALUABLE

- `PASS`：满足 8.3 全部条件。
- `FAIL`：主 skill `< 0`；或发生非有限预测、非正 scale、quantile crossing；或第 5–13 项
  任一科学护栏越界；或发现泄漏、hash drift、formal 后调参、选择性重跑。
- `INCONCLUSIVE`：主 skill `>= 0`，但未达到 2%、CI 下界不大于 0、3/5 数据 seed 规则未达成
  或 2/3 initialization 规则未达成，同时不存在 FAIL 条件。
- `NOT_EVALUABLE`：硬件、I/O、租期、进程、容器或网络等运行故障导致 120/120 不完整，且
  没有科学完整性违规。它是操作失败，不解释为科学 FAIL。

`INCONCLUSIVE` 与 `NOT_EVALUABLE` 都阻断 Stage 3/4。增加证据必须走新的 CCP 和新的 sealed
formal seeds；不能降低本文件门槛，不能把旧 formal test 当作 validation。

## 9. 代码架构与拟新增文件

实施沿用现有 contracts、Stage1B predictor/training/checkpoint、execution scheduler/state、E01
bundle/supervision/freeze 模式，不复制第二套跨阶段契约。

拟新增或扩展的主要边界：

```text
configs/stage2/stage2_v1.yaml
configs/e02/e02_v1.yaml
src/tarca/stage2/
  config.py
  seeds.py
  data.py
  baselines.py
  dlinear.py
  neural.py
  distributions.py
  training.py
  selection.py
  manifest.py
  runner.py
  freeze.py
src/tarca/e02/
  config.py
  grant.py
  scoring.py
  bootstrap.py
  decision.py
  runner.py
  receipt.py
scripts/run_stage2_v1.py
scripts/run_e02_v1.py
scripts/prepare_stage2_v1_server_bundle.py
deploy/stage2/Dockerfile
deploy/stage2/compose.yaml
deploy/stage2/entrypoint.sh
deploy/stage2/bootstrap.sh
deploy/stage2/supervisor.sh
deploy/stage2/requirements-server.lock
tests/stage2/
tests/e02/
```

文件可以在不改变模块职责的前提下进一步拆小。任何公共 contract 变更必须先证明现有 schema
无法表达需求并走 CCP；本实施默认只加 adapter，不修改 Stage1A 公共 schema。

## 10. CLI 与状态机

固定入口：

```text
python scripts/run_stage2_v1.py prepare
python scripts/run_stage2_v1.py dry-run
python scripts/run_stage2_v1.py preflight
python scripts/run_stage2_v1.py launch
python scripts/run_stage2_v1.py resume
python scripts/run_stage2_v1.py status
python scripts/run_stage2_v1.py freeze
python scripts/run_stage2_v1.py recover

python scripts/run_e02_v1.py prepare
python scripts/run_e02_v1.py dry-run
python scripts/run_e02_v1.py preflight
python scripts/run_e02_v1.py launch
python scripts/run_e02_v1.py resume
python scripts/run_e02_v1.py status
python scripts/run_e02_v1.py finalize
python scripts/run_e02_v1.py recover
```

`prepare/dry-run/status` 不授权服务器写入或 formal access。Stage 2 正式训练 launch 需要用户在
对应授权消息中提供精确确认串：

```text
I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN
```

E02 formal launch 需要另一个精确确认串：

```text
I_ACKNOWLEDGE_E02_V1_FORMAL_RUN
```

确认串只是代码层二次保护，不替代服务器 runbook 所要求的动作范围授权。当前用户对方案 B
和本地实施的确认不自动授权连接、上传、启动容器、产生费用或访问 formal 数据。

task 状态机：

```text
PLANNED -> READY -> STARTING -> RUNNING -> SUCCEEDED
                                  |  |  |
                                  |  |  +-> STALLED -> RETRYING -> RUNNING
                                  |  +----> FAILED
                                  +-------> CANCELLED_BY_ENV_RESET
```

只有校验通过的 artifact 才能使 task 进入 `SUCCEEDED`。运行账本不是科学结果本身。

## 11. 双 RTX 4090 与 28 核 CPU 调度

### 11.1 资源预算

服务器的硬准入预算固定为：

| 资源 | 总量 | TARCA 可调度 | 保留 |
|---|---:|---:|---:|
| CPU | 28 | 24 | 1 核 scheduler/monitor + 3 核系统/I/O |
| RAM | 224GB | 200GiB admission ceiling | 至少 24GiB 系统余量 |
| GPU | 2 × 4090 24GB | 2 个独占 GPU worker | 每卡避免超过 20GiB 声明显存 |
| 本地高速存储 | preflight 实测 | 最低 200GiB free | 推荐 300GiB free NVMe |

每个大型 GPU training task 初始声明 `4 CPU threads + <=32GiB RAM + 1 GPU + <=20GiB VRAM`。
两卡同时训练时使用 8 个 CPU 核和最多 64GiB RAM；剩余 16 个可调度 CPU 核运行下一批数据
生成、VAR/DLinear、cross-fitting、预测后处理和上一批评分。所有进程设置 CPU affinity，并
限制 OMP/MKL/OpenBLAS 线程，防止 24 核预算被进程乘法过量订阅。

### 11.2 有用饱和流水线

执行图按以下方式重叠：

```text
首批 TRAIN/VALIDATION shards 完成
          |
          +--> GPU0: neural task A ----> inference bundle A
          +--> GPU1: neural task B ----> inference bundle B
          +--> CPU: 生成剩余 shards / VAR / DLinear cross-fit
                                      |
                                      +--> CPU score 已完成预测
                                      +--> GPU0/GPU1 接下一训练任务
                                                       |
                                                       +--> 最终聚合/冻结
```

六个大型训练任务是 PatchTST 三个 initialization 与 iTransformer 三个 initialization。默认
GPU0/GPU1 各运行一个独立训练任务，完成即从同一固定队列领取下一项。只要依赖已满足，
不让一张卡等待另一张卡完成同一轮。

CPU 队列以 backpressure 驱动：优先保证两张 GPU 不因数据等待空闲，再使用剩余核生成后续
数据和运行线性任务，最后运行可延迟的 scoring/bootstrap。若 RAM 或 I/O 接近上限，暂停
低优先级 CPU producer，不改变科学任务。

### 11.3 独占 GPU worker 与推理装箱

训练阶段每卡保持一个独占 worker，不启动多个互不知情的 CUDA 训练进程争抢显存。小型
validation/formal inference 在同一独占 worker 内合并 2–3 个不可变 bundle：

- 从 1 个 bundle 开始，连续 180 秒 GPU util `<70%` 且显存 `<8GiB` 时试 2 个；
- 继续连续 180 秒 util `<80%` 且总显存 `<18GiB` 时试 3 个；
- 显存 `>20GiB`、OOM、thermal throttle、data-wait 增大或单位样本吞吐下降时退回；
- OOM 只允许降低同卡 bundle 并自动重试一次；
- 装箱不得改变 batch size、样本、模型、precision 或输出顺序，artifact identity 保持不变。

### 11.4 DDP

4090 之间无 NVLink，默认采用任务级并行。只有固定代表性 iTransformer probe 显示双卡 DDP
相对单卡端到端墙钟时间缩短至少 `30%`，且数值一致性、显存和 checkpoint/reload 都通过，
才允许让单个正式任务使用 DDP。否则两卡分别执行不同 initialization。probe 的模型、batch、
数据规模和判定脚本进入 manifest，不能凭瞬时 utilization 选择 DDP。

### 11.5 启动 ETA gate

preflight 用最小代表性数据、一个 linear fit、两个 neural forward/backward/checkpoint/reload
和固定步数训练估计 critical path。只有：

```text
estimated_completion + 1 hour safety margin < remaining rental window
```

且存储、RAM、I/O、两卡健康全部通过，formal launch 才能继续。预计超时必须停止并报告，
不得静默缩减科学工作量。

## 12. 服务器 bundle 与一键入口

本地实施最终生成内容寻址、可离线校验的 server bundle，包含：

- 当前 Git commit 的受控源码快照；
- Stage 2/E02 配置和 schema；
- 四个固定外部源码 capsule 及 hashes；
- server dependency lock，不覆盖镜像自带 CUDA PyTorch；
- Dockerfile、compose、entrypoint、bootstrap、supervisor；
- tests/smoke、preflight 和 recovery 工具；
- bundle manifest、SBOM、文件哈希、构建时间和不含秘密的环境说明。

本地构建入口固定为：

```text
python scripts/prepare_stage2_v1_server_bundle.py --output artifacts/stage2/server_bundle
```

服务器 bundle 解压后唯一 bootstrap 入口固定为：

```text
bash deploy/stage2/bootstrap.sh --mode preflight
```

bootstrap 只做环境、镜像、依赖、GPU、存储、来源和 manifest 校验，并输出下一条建议命令；
没有相应确认串时不启动 Stage 2 正式训练或 E02。服务器凭据、SSH 目标、token 和秘密不写入
bundle、配置、日志或 Git。

容器必须验证 Python 3.10、PyTorch 2.2.2、CUDA 12.1、cuDNN、两张可见 4090、GPU UUID、
驱动兼容、AMP、atomic checkpoint 和 reload。requirements lock 不得安装另一个 torch wheel
覆盖基础镜像。

## 13. 24 小时环境重置与恢复

服务器运行按“环境可能在 24 小时被重置”设计：

- 每个 epoch 至少一次原子 checkpoint；
- task 完成时先写临时目录，校验后原子 rename；
- checkpoint 绑定 task/science/config/data/model/precision hashes；
- SQLite WAL 记录 run、task、attempt、heartbeat、resource sample、alert 和 artifact ref；
- scheduler 重启先核对 PID、GPU PID、checkpoint 和 artifact，不能重复启动存活任务；
- 已成功且 artifact 验证通过的任务不重跑；
- 暂时性 worker/I/O 错误最多自动重试一次；
- hash mismatch、truth leakage、NaN、partition drift、formal access 越权直接 fail closed；
- 正式 trajectory 的成功和失败 attempt 全保留，不允许只重跑不利 scientific unit。

`recover` 生成可上传的 recovery capsule，包含最新一致 checkpoint、运行数据库快照、manifest、
日志索引和 hashes。它不包含凭据，不把未完成科学分数写入公开摘要。环境恢复后先执行
`preflight`，再用同一 run identity `resume`；不得创建一个看似全新的 formal run 规避失败
历史。

## 14. 只读监控与科学盲法

沿用现有只读监控平面，展示：

- run identity、状态、总 task 数和完成进度；
- 每个 worker 的 PID、heartbeat、attempt、CPU affinity、RSS/PSS；
- GPU0/GPU1 的任务、utilization、显存、功耗、温度和 data-wait；
- 磁盘空间、读写吞吐、队列长度、ETA 和告警；
- 期望资源与实际资源的差异。

正式 Stage 2/E02 完成前，监控/API/日志摘要禁止展示 partial CRPS、NLL、MAE、coverage、
seen/unseen 差异、模型排名、最佳 initialization 或 PASS 倾向。监控只读，不提供暂停、kill、
重跑、改配置、改 seed 或改并发按钮。调度器只读取资源遥测，不读取科学 score 做 placement。

## 15. 安全与授权边界

服务器访问严格服从 `docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md`：

- 当前本地实施不隐含任何远程访问授权；
- 后续连接、上传、构建、启动和 formal access 分别限制在用户明确授权的范围内；
- 不改变服务器凭据、防火墙、Docker daemon、SSH 配置或第三方资源；
- 服务端口只绑定可信本地接口，监控通过用户批准的访问方式查看；
- 不在命令行、日志、artifact、Git 或监控中打印秘密；
- bundle 接收的路径、配置和确认串全部 schema 校验；
- 容器不挂载 Docker socket，不使用特权模式，不开放公网控制面。

## 16. TDD、测试与覆盖率

实现使用 RED → GREEN → REFACTOR。至少覆盖：

### 16.1 单元测试

- seed namespace、碰撞拒绝和 formal seed 隔离；
- split、normalizer TRAIN-only、窗口不跨边界；
- Last/Seasonal/VAR/DLinear mean 与 training-only scale；
- Gaussian shape、finite、positive scale、de-normalization；
- CRPS/NLL/MAE、coverage、skill 和 decision 边界；
- 5000 次 stratified paired trajectory bootstrap 的确定性；
- 2%、3/5、2/3、5%、10% 等所有等号边界；
- manifest/hash、checkpoint、grant 和 formal access 拒绝；
- resource admission、GPU packing、DDP 30% gate 和 ETA gate。

### 16.2 集成测试

- 小型 Stage 2 prepare → train → select → freeze → reload；
- 两个 GPU worker 的 fake-NVML 调度与 CPU backpressure；
- worker crash、OOM 降并发、scheduler restart 和 atomic resume；
- E02 在 synthetic fixed fixture 上分别得到 PASS/FAIL/INCONCLUSIVE/NOT_EVALUABLE；
- monitor/API 不泄露科学分数；
- server bundle 离线 hash 校验和 bootstrap dry-run；
- Python 3.10 import、官方 wrapper 最小 forward/backward/checkpoint/reload。

### 16.3 GPU 与容器 smoke

本地无双 4090 时只运行 CPU/fake-NVML 测试，不伪称 GPU 验收通过。服务器 preflight 必须让
两张卡分别完成 CUDA tensor、AMP、一个 PatchTST/iTransformer step、checkpoint/reload 和
并发代表性 probe。

### 16.4 质量门槛

- 全项目已有测试不得回归；
- 新增 Stage 2/E02 代码 branch coverage 不低于 `80%`；
- Ruff、mypy 和 schema validation 通过；
- dependency/security audit 无未解释 critical/high 问题；
- `git diff` 检查无 hard-coded secret、formal artifact 和无关用户改动。

## 17. 验收标准

本地实施只有同时满足以下条件才算完成：

1. 方案 B 的数据、模型、概率、门槛和结论规则全部进入机器可校验配置；
2. Stage 2/E02 science manifests 可确定性编译且哈希稳定；
3. 六类 predictor 通过最小训练/预测/reload，概率输出合同有效；
4. formal seeds 在未授权入口中不可达；
5. Stage 2 freeze 与 E02 grant/finalize 严格分离；
6. 双 GPU task queue、独占 worker、CPU backfill、动态 inference bundle、DDP gate 可测试；
7. 资源不足、ETA 超时和 source drift 都 fail closed，不静默降配；
8. checkpoint、SQLite ledger、recovery capsule 和 resume 可从中断恢复；
9. 只读监控展示资源和进度但不泄露中间科学结论；
10. server bundle 在目标镜像上有单一 preflight 入口，无需在线补源码或临时改配置；
11. 测试、覆盖率、lint、typecheck、安全扫描和 bundle hash 校验通过；
12. 本地实现过程中没有连接服务器、没有运行完整 Stage 2/E02、没有访问 formal 数据；
13. 实施报告明确列出本地已验证项与必须在双 4090 服务器验证的项；
14. 用户后续可通过独立授权让服务器立即进入 preflight 和正式运行流程。

## 18. 实施顺序

1. 把本设计转换成逐文件 TDD 实施计划；
2. 先写配置、seed、manifest、grant 和 decision 的失败测试；
3. 实现数据 bridge、TRAIN-only normalizer 和 baseline 概率输出；
4. 接入固定 DLinear、PatchTST、iTransformer source capsule 与 wrappers；
5. 实现训练、checkpoint、validation selection 和 Stage 2 freeze；
6. 实现 E02 scoring、bootstrap、decision、receipt 和 formal access boundary；
7. 扩展 scheduler、独占 GPU worker、CPU backfill、packing/DDP/ETA gates；
8. 构建 Docker、bootstrap、supervisor、recovery 和离线 server bundle；
9. 完成单元、集成、fake-NVML、Python 3.10 和本地 smoke；
10. 做代码、安全、科学规则映射和无 formal access 的最终复核；
11. 输出本地实施报告和服务器首开命令；
12. 等待用户另行授权服务器连接、上传、preflight、Stage 2 training 和 E02 formal run。

## 19. 设计决策的合理性

- `2%` 主 skill 门槛高于“只要正数就算成功”，但低于把工程上可用的小幅提升误设为无法
  达成的高门槛；区间下界和 seed 多数规则同时限制偶然性。
- 120 条完整 formal trajectory 与分层配对 bootstrap 利用同场景配对优势，同时避免把大量
  重叠窗口错误当作独立样本夸大显著性。
- 3/5 data seeds 和 2/3 initialization 把“数据偶然性”与“训练偶然性”分开；保留
  `INCONCLUSIVE` 防止证据不足被强行解释成理论失败。
- 最强线性基线在 validation 预选，避免只与弱 naive 比；iTransformer 预先固定，避免 formal
  test 后在多个神经模型中择优。
- calibration、NLL、MAE 和远时距护栏防止模型只优化短期 CRPS，却输出失真的概率尺度或在
  其他关键维度严重退化。
- 双 GPU 采用独立训练任务并行，符合无 NVLink 的 4090；小推理在独占 worker 内装箱，兼顾
  利用率、显存安全和可恢复性。
- 24 核工作预算和 200GiB RAM ceiling 给操作系统、I/O 与监控留出余量，剩余 CPU 与 GPU
  流水线重叠，目标是缩短 critical path，而不是制造表面 100% 占用。

## 20. 明确冻结的结论

方案 B 的科学规则在代码实施期间不得因开发 smoke、validation 或服务器性能探针结果降低。
若实现发现协议矛盾、资源不够或正式 ETA 超出租期，应停止并报告，通过新的书面变更决定；
不得自行删模型、减 seed、减 trajectory、缩 horizon、降低 2% 门槛或放宽校准护栏。

本文件通过书面复核后才进入实施计划和代码阶段。代码完成仍只代表“服务器就绪”，不代表
Stage 2 已训练完成或 E02 已通过。
