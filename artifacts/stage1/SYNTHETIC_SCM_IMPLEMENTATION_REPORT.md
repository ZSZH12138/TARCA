# TARCA Stage 1B Synthetic Regime-Switching SCM 实施报告

> 报告日期：2026-07-26
> 分支：`codex/stage1-synthetic-scm`
> 实施基线：`f376d60ccf2437494b10e04b3ce98eeebfed9b88`
> 报告前代码提交：`fceb9aa59f46f8882dc4f80822c7498cf194a850`
> 研究状态：`ENGINEERING_SMOKE_ONLY`

## 第一部分：权威顺序、前置理解与阻断审计

### 1.1 权威顺序

本次实施严格按以下顺序解释冲突与范围：

1. 冻结的预注册、假设账本、创新声明和术语；
2. 已冻结的 Stage 0 边界与已测试的 Stage 1A 统一数据契约；
3. `docs/TARCA_项目计划书.md`；
4. `docs/TARCA_具体实施计划.md`；
5. 用户上传的
   `D:\TARCA_Stage1_合成_Regime_Switching_SCM.md`，SHA-256 为
   `5a1bcf2263c77afe253df7dd9567d2b074bbc086bdf56aa975467285acffc01a`。

用户明确授权了两项最小兼容修复：

- Stage 1 范围测试仅放行 `src/tarca/data/synthetic/`，仍拒绝预测器、机制定位、
  内部干预、OT、DAS、DRO、金融和其他后续模块；
- 不修改统一 `DataManifest` 契约；规范 TEST 汇总两个物理测试分区，
  合成专属的四分区信息、真值和生成溯源保存在严格私有 sidecar 中。

用户另明确豁免了“当前可用内存至少 6 GiB”这一项资源门禁；未豁免 CPU-only、
磁盘、依赖、范围、测试或科研表述门禁。

### 1.2 前置理解

- Stage 0 已冻结。本任务没有提交任何 `src/tarca/stage0/`、依赖锁或 Stage 0
  报告改动。
- Stage 1A 统一数据契约已存在且可导入；本实现复用 `WindowBatch`、
  `DataManifest`、`DataSplitSummary`、`SplitPartition` 和 `InterventionPair`。
- 本任务只建立人工合成数据生成过程、模型计算层面的概念干预真值和工程 smoke。
- 合成真值只对本人工 SCM 成立，不支持真实市场或金融因果结论。

### 1.3 阻断审计

| 阻断项 | 结论 | 证据 |
|---|---|---|
| 统一数据契约缺失 | 未触发 | 428 个契约行为测试通过 |
| Stage 0 基线失败 | 未触发 | Doctor PASS；全仓库 994 通过、2 跳过 |
| 无法兼容的契约冲突 | 未触发 | 两项冲突均按用户授权的最小兼容方案解决 |
| 新增重依赖/复制第三方代码 | 未触发 | `uv.lock` 未改变；仅使用现有依赖；无复制实现 |
| 本地资源不足 | RAM 项已由用户豁免，其余通过 | 12 逻辑核、约 169 GiB 可用磁盘、CPU PyTorch |
| 目标文件存在重叠修改 | 未触发 | 实施目标文件开始时均无用户修改 |

实施前已有的 7 个用户工作树改动始终未暂存、未提交：

- `README.md`
- `artifacts/stage0/STAGE0_IMPLEMENTATION_REPORT.md`
- `docs/TARCA_具体实施计划.md`
- `docs/TARCA_项目汇报书.md`（用户工作树中为删除）
- `docs/TARCA_项目计划书.md`
- `docs/stage0_scope.md`
- `tests/test_operator_docs.py`

## 第二部分：资源结论（对应提示词 17.1）

```text
RESOURCE_DECISION: LOCAL_CPU_CONTINUE
```

| 项目 | 实测/估算 | 判断 |
|---|---:|---|
| 逻辑/物理 CPU | 12 / 6 | 通过至少 2 个逻辑核的门槛 |
| 总内存 | 16,915,316,736 bytes | 机器总量满足推荐范围 |
| Doctor 时可用内存 | 4,232,732,672 bytes | 低于 6 GiB 硬门槛；用户已明确豁免 |
| smoke 时可用内存 | 4,299,186,176 bytes | 串行执行，无 OOM |
| 磁盘可用 | 181,288,751,104 bytes | 通过 |
| CUDA/GPU | 不可用；`gpu_used=false` | 本阶段不要求 GPU |
| easy smoke 附加内存估算 | 26,580,672 bytes | 低于 4 GiB 上限 |
| 持久化数据大小 | 20,886,690 bytes | 低于 2 GiB 上限 |
| smoke 完整目录大小 | 20,891,756 bytes | 含 JSON/Markdown 证据 |
| smoke 运行时间 | 20.632785 s | 低于 30 分钟上限 |

内存数值采用“truth 数组 + `WindowBatch` tensors + analytic MC 数组”的确定性字节
核算并乘 2 倍安全系数，不声称是操作系统级峰值实测。

## 第三部分：实施内容与科研状态（对应提示词 17.2）

```text
IMPLEMENTATION_STATUS: COMPLETED_STAGE1B_ENGINEERING
SCIENTIFIC_STATUS: ENGINEERING_SMOKE_ONLY
```

已完成：

- 严格、冻结、拒绝额外字段的 easy/medium/hard 配置；
- 单一根 `SeedSequence` 派生的具名随机流；
- regime 转移、稳定非线性 VAR、趋势/尺度潜变量、冲击和外生输入；
- seen/unseen regime 的同标签参数位移；
- 显式 future noise bank 和 paired factual/counterfactual replay；
- trend/scale 概念隔离、source=base/no-intervention 精确零效应；
- MC 均值、标准差和分位数干预效应；
- none、MCAR 和整向量 block missingness；
- 连续 `60/20/10/10` 物理切分、不跨边界窗口、train-only normalization；
- 规范统一 manifest、合成 provenance、私有 Arrow、NPZ 和 checksum 持久化；
- 独立复算的真值、重放、hash、split、scaler、manifest、Arrow 和安全验证；
- 固定 easy 配置的 CPU-only E01 工程 smoke；
- 安全的构建 CLI 与 smoke CLI。

未把工程结果解释为正式 E01 统计结论、Gate A 结论或 TARCA 科学假设验证。

## 第四部分：内容与接口映射（对应提示词 17.3）

| 功能 | 实际接口/模块 | 与其他模块的连接 |
|---|---|---|
| 配置与数据集 | `SyntheticConfig`、`SyntheticDataset`、`load_synthetic_config`、`build_synthetic_dataset` | 驱动所有生成模块并输出统一契约批次 |
| regime 与随机流 | `regimes.py` | 为潜变量、动力学、缺失机制分配独立确定性流 |
| 潜变量 | `LatentConceptPath`、`generate_latent_concepts`、`replace_concept_at_origin` | 向 SCM rollout 和 oracle 提供 trend/scale 真值 |
| 非线性 SCM | `RegimeDynamics`、`generate_regime_dynamics`、`rollout_nonlinear_var` | 消费 regime、概念、外生输入、噪声和冲击 |
| paired oracle | `FutureNoiseBank`、`replay_paired_counterfactual`、`monte_carlo_oracle` | factual/counterfactual 共享全部未来随机量 |
| missingness | `generate_missing_mask`、`apply_missingness` | 生成未来不可见、`True=observed` 的 mask |
| 时间切分 | `PhysicalSplit` 和 builder 私有窗口化 | 生成 train/validation/test-seen/test-unseen |
| 统一数据契约 | `WindowBatch`、`DataManifest` | 四个物理 split 映射到规范 TRAIN/VALIDATION/TEST |
| 私有持久化 | `persist_synthetic_dataset`、20 字段 Arrow schema | 输出固定九文件并以 checksum 和 dataset hash 连结 |
| 独立验证 | `validate_synthetic_dataset` | 不调用 builder 私有验证器，独立复算身份与语义 |
| E01 smoke | `run_e01_engineering_smoke` | 复用真实 easy 数据验证，并运行固定 analytic controls |
| CLI | 两个 `scripts/*.py` | 仅负责编排、路径门禁、原子发布和退出码 |

关键语义：

- `true_delay=δ` 与预测 horizon `h` 分离；easy case 的首个效应峰值满足 `h=δ+1`；
- `WindowBatch.regime` 是 prediction origin 前最后一个历史步的 regime；
- 规范 TEST 的 count 是两个物理测试块之和，hash 是二者固定映射的规范 JSON hash；
- Arrow、窗口 ID 和 dataset hash 的依赖方向避免循环身份；
- `truth.npz` 只含数值数组，读取必须 `allow_pickle=False`。

## 第五部分：文件变更（对应提示词 17.4）

### 5.1 配置与文档

| 文件 | 职责 |
|---|---|
| `.gitignore` | 仅忽略 `artifacts/stage1/synthetic_scm_smoke/` 生成证据 |
| `configs/synthetic/synthetic_easy.yaml` | 冻结 easy 配置 |
| `configs/synthetic/synthetic_medium.yaml` | 冻结 medium 配置 |
| `configs/synthetic/synthetic_hard.yaml` | 冻结 hard 配置 |
| `docs/stage1_synthetic_scm.md` | 设计、契约映射、来源与许可证边界 |
| `artifacts/stage1/SYNTHETIC_SCM_IMPLEMENTATION_REPORT.md` | 本十部分实施报告 |

### 5.2 源码与命令

| 文件 | 职责 |
|---|---|
| `src/tarca/data/__init__.py` | Stage 1 数据包边界 |
| `src/tarca/data/synthetic/__init__.py` | 合成模块公共导出 |
| `src/tarca/data/synthetic/regimes.py` | 配置基础、随机流、regime 和参数位移 |
| `src/tarca/data/synthetic/latent_concepts.py` | trend/scale 生成和 origin 干预 |
| `src/tarca/data/synthetic/nonlinear_var.py` | 稳定 regime-switching nonlinear VAR |
| `src/tarca/data/synthetic/counterfactual_oracle.py` | paired replay、MC 效应和 delay 估计 |
| `src/tarca/data/synthetic/missingness.py` | none/MCAR/block 缺失机制 |
| `src/tarca/data/synthetic/dataset_builder.py` | 组合、切分、标准化、manifest、Arrow 和原子持久化 |
| `src/tarca/data/synthetic/validation.py` | 公共验证与 E01 工程 smoke |
| `src/tarca/data/synthetic/_validation_core.py` | 严格报告记录、真值与编排验证 |
| `src/tarca/data/synthetic/_validation_integrity.py` | 独立 split/hash/manifest/oracle 完整性验证 |
| `src/tarca/data/synthetic/_validation_persistence.py` | 独立九文件、NPZ、Arrow 和 checksum 验证 |
| `scripts/build_synthetic_dataset.py` | 安全构建、seed override、验证和原子发布 CLI |
| `scripts/run_synthetic_oracle_smoke.py` | CPU-only smoke 与 JSON/Markdown 原子证据 CLI |

### 5.3 测试

| 文件 | 职责 |
|---|---|
| `tests/contracts/test_stage1_scope.py` | 最小放行 Stage 1B 合成模块并继续禁止后续范围 |
| `tests/data/synthetic/test_regimes.py` | 配置、随机流、转移与 unseen 参数测试 |
| `tests/data/synthetic/test_latent_concepts.py` | 潜变量递推、隔离、不变性与干预测试 |
| `tests/data/synthetic/test_nonlinear_var.py` | 稳定性、lag、rollout 和 analytic case 测试 |
| `tests/data/synthetic/test_counterfactual_oracle.py` | paired noise、效应、MC 和 delay 测试 |
| `tests/data/synthetic/test_missingness.py` | mask、未来不可见性和 block union 测试 |
| `tests/data/synthetic/test_dataset_builder.py` | split/scaler/manifest/hash/Arrow/安全持久化测试 |
| `tests/data/synthetic/test_validation.py` | 独立真值与 E01 验证、负对照和伪造检测 |
| `tests/data/synthetic/test_cli.py` | CLI、路径、复现、CPU、原子发布和失败清理测试 |

### 5.4 生成但不跟踪的 smoke 证据

`artifacts/stage1/synthetic_scm_smoke/` 已生成并由窄规则忽略，其中包含：

- `dataset/` 下固定九个数据文件；
- `e01_engineering_smoke.json`；
- `e01_engineering_smoke.md`。

没有生成数据、缓存、模型权重或运行日志被加入 Git。

## 第六部分：TDD、测试与命令证据（对应提示词 17.5）

### 6.1 TDD 证据

各生产行为均先出现真实 RED，再完成最小 GREEN。代表性 RED 包括：

- 模块不存在与接口缺失；
- 非整数参数被分数位移；
- 潜变量递推或 provenance 可伪造；
- oracle 不能支持每个 bank 的独立 regime path；
- overlapping block missingness；
- dataset replay 初始历史缺失；
- 验证器依赖 builder 私有函数、配置伪装 easy、hash/Arrow/reparse 伪造；
- scale 干预污染 trend、非平凡干预被静默忽略；
- CLI 持久化验证失败遗留输出；
- staging 同名替换导致清理越界、staging 创建失败泄漏裸异常。

所有 Critical/High 审查发现均在提交前以回归测试关闭。

### 6.2 最终命令矩阵

| 实际命令（摘要） | 退出码 | 结果 |
|---|---:|---|
| `uv lock --check` | 0 | 128 个包解析，锁文件未改 |
| `python scripts/doctor.py` | 0 | 总状态 PASS |
| `python -m compileall -q src scripts tests` | 0 | PASS |
| `ruff check .` | 0 | PASS |
| `ruff format --check .` | 0 | 75 个文件已格式化 |
| `pytest tests/contracts -q`（原样 addopts） | 1 | 428 行为测试通过；仅因未导入 `tarca.stage0`，冻结 coverage 得到 0% |
| `pytest tests/contracts -q -o "... --no-cov"` | 0 | 428 通过 |
| `pytest tests/data/synthetic ... --no-cov` | 0 | 403 通过、1 跳过（安全修复前全集） |
| `pytest tests/data/synthetic --cov=tarca.data.synthetic` | 0 | 406 通过、1 跳过；89.87% |
| `pytest -q` | 0 | 994 通过、2 跳过；Stage 0 coverage 91.17% |
| `pre_commit run --files <Stage1B files>` 第一次 | 1 | 发现 2 个 lint 问题，无安全 hook 失败 |
| 同一 file-scoped pre-commit 修复后重跑 | 0 | 全部 hooks PASS |
| lint 修复后的相关 record/resource 测试 | 0 | 9 通过；可用内存查询通过 |
| `build_synthetic_dataset.py` easy，同 seed 两次 | 0 / 0 | 两次 `BUILD_PASS`，dataset hash 相同 |
| 同 CLI 使用 seed `20260726` | 0 | `BUILD_PASS`，dataset hash 改变 |
| `run_synthetic_oracle_smoke.py` easy | 0 | `ENGINEERING_SMOKE_PASS` |

Windows 符号链接负例因当前账户不能创建测试符号链接而跳过；reparse/交换攻击仍由
可运行的目录身份测试和 builder 安全测试覆盖。全仓库第二个 skip 是既有的 Windows
平台条件 skip。

### 6.3 复现证据

相同 config/seed 两次得到：

```text
config_hash:
sha256:0236e94ac2ff6ef9523fd86f80c5c576278dd0817f36cb40a4cc0c839f40601c

dataset_hash:
sha256:63b1efa02cf6b5a5848dfad0fea62e9ead2ae82b3340cf8a31a4e7a5768b430b
```

`config_resolved.yaml`、`normalization.json`、`truth.npz` 和四个 Arrow 文件逐字节一致。
`manifest.json` 因真实 UTC `generated_at` 不同而不要求逐字节一致；
`checksums.json` 相应不同。时间字段不进入 dataset identity。

seed 改为 `20260726` 后：

```text
config_hash:
sha256:eb58d68a7370966e5573200f82717dbd538cf457e6a36e4733a5968a2bcb8353

dataset_hash:
sha256:d79f07eebf4cb938dbe1b1c34e8ea310a0bf24153263aa6744716566f74ff459
```

## 第七部分：E01 工程 smoke 结果（对应提示词 17.6）

```text
STATUS: ENGINEERING_SMOKE_PASS
RESEARCH_STATUS: ENGINEERING_SMOKE_ONLY
```

配置：

- `synthetic_easy`
- root seed：`20260725`
- pair count：`16`
- MC sizes：`[32, 64, 128, 256]`
- config hash：
  `sha256:0236e94ac2ff6ef9523fd86f80c5c576278dd0817f36cb40a4cc0c839f40601c`
- data hash：
  `sha256:63b1efa02cf6b5a5848dfad0fea62e9ead2ae82b3340cf8a31a4e7a5768b430b`

最终误差：

| 指标 | 数值 |
|---|---:|
| trend mean RMSE | 0.00000000 |
| scale mean RMSE | 0.03182624 |
| scale std relative error | 0.04882564 |
| conditional variance relative error | 0.09741550 |
| quantile normalized RMSE | 0.10122186 |
| estimator variance | 0.00094034 |
| convergence log-error slope | -0.36575645 |

延迟恢复：

```text
true_delay δ = 2
estimated_delay δ = 2
absolute_error = 0
首个预期效应 horizon h = δ + 1 = 3
```

对照签名距离：

| 模型/控制 | 距离 |
|---|---:|
| correct SCM | 0.00463353 |
| wrong delay | 0.56570440 |
| wrong scale | 0.25672288 |
| random concept | 1.20026004 |

MC 收敛：

| Samples | Total error | Estimator variance |
|---:|---:|---:|
| 32 | 0.16806193 | 0.00369625 |
| 64 | 0.10661806 | 0.00357955 |
| 128 | 0.09119236 | 0.00154729 |
| 256 | 0.07604686 | 0.00094034 |

运行事实：

- runtime：20.632785 s；
- additional memory estimate：26,580,672 bytes；
- dataset output：20,886,690 bytes；
- logical CPUs：12；
- GPU available：false；
- GPU used：false；
- 网络请求、数据下载、模型训练：均未执行。

## 第八部分：15–30 分钟人工核对步骤（对应提示词 17.7）

以下命令均在仓库根目录 PowerShell 中执行，并使用指定 Conda 环境的绝对解释器：

```powershell
$python = 'D:\software\MyAnaconda\envs\tarca-stage0\python.exe'
$uv = 'D:\software\MyAnaconda\envs\tarca-stage0\Scripts\uv.exe'
$run = ".pytest_cache/manual-stage1-$([guid]::NewGuid().ToString('N'))"

& $uv lock --check
& $python scripts/doctor.py
& $python -m pytest tests/data/synthetic -q `
  -o 'addopts=--strict-config --strict-markers --no-cov'

& $python scripts/build_synthetic_dataset.py `
  --config configs/synthetic/synthetic_easy.yaml `
  --output "$run/same-a" --smoke
& $python scripts/build_synthetic_dataset.py `
  --config configs/synthetic/synthetic_easy.yaml `
  --output "$run/same-b" --smoke
& $python scripts/build_synthetic_dataset.py `
  --config configs/synthetic/synthetic_easy.yaml `
  --output "$run/changed-seed" --seed 20260726

& $python scripts/run_synthetic_oracle_smoke.py `
  --config configs/synthetic/synthetic_easy.yaml `
  --output "$run/oracle-smoke"

& $python -c "import json,sys; a=json.load(open(sys.argv[1])); print(a['status'], a['validation']['status'], a['true_delay'], a['estimated_delay'], a['correct_signature_distance'], a['wrong_delay_signature_distance'], a['wrong_scale_signature_distance'], a['random_concept_signature_distance'], a['gpu_used'])" `
  "$run/oracle-smoke/e01_engineering_smoke.json"

git status --short
git status --short --ignored
```

人工 PASS 标准：

1. lock 与 Doctor 返回 0；
2. synthetic targeted tests 全部通过，最多只出现说明过的 Windows symlink skip；
3. 两次相同配置输出相同 `config_hash` 和 `dataset_hash`；
4. changed-seed 的两个 hash 与原值不同；
5. `manifest.json` 中有转移矩阵、谱半径、resolved delay、四个物理 split；
6. `normalization.json` 的 fit interval 只覆盖 train；
7. smoke JSON 的 validation 为 `VALIDATION_PASS`，issues 为空；
8. no-intervention/source=base 与 paired-noise/concept-isolation 已由
   `oracle_invariants` PASS 覆盖；
9. true delay 与 estimated delay 都为 2，预测步解释为 `h=δ+1=3`；
10. correct distance 小于 wrong-delay、wrong-scale、random-concept；
11. `gpu_used=false`，命令没有联网、下载或训练；
12. `git status` 不出现数据、缓存、日志、密钥或后续阶段模块被跟踪。

## 第九部分：限制、外部来源与未实现边界（对应提示词 17.8）

- 本阶段没有训练预测器；
- 没有实现 `ForecastDistribution` 的基础预测模型；
- 没有实现内部激活捕获或内部表示干预；
- 没有实现 PLOT、OT、DAS、DRO、机制定位或鲁棒性训练；
- 没有进入真实数据、金融数据或真实世界因果推断；
- 非线性多步 SCM 不声称存在一般闭式干预效应；
- E01 仅为固定 easy case 的工程 smoke；正式阈值、统计协议和 sweep 尚需冻结；
- medium/hard 配置通过配置与单元/组合测试，但没有在本次最终门禁中执行完整持久化
  和正式 sweep；
- NumPy truth 数组是真只读；PyTorch Tensor 没有底层只读类型，当前只保证独立
  owned tensors 和 frozen wrapper；
- Windows 发布提供同卷 rename 的原子可见性，不声称具备断电条件下目录 fsync 的
  完整耐久保证；
- 文献仅用于设计参考，详见 `docs/stage1_synthetic_scm.md`；没有复制第三方源码；
- 未调用外部 LLM、网络、下载器或新增依赖，`uv.lock` 保持不变；
- 合成 SCM 因果真值只对该人工数据生成过程成立。

## 第十部分：下一阶段入口（对应提示词 17.9）

```text
冻结 E01 正式协议与阈值
→ 审核并生成正式 synthetic 数据
→ 实现基础预测器和统一 ForecastDistribution
```

在以上前置完成前，不进入机制定位、内部激活干预、PLOT/OT/DAS/DRO 或金融实验。
