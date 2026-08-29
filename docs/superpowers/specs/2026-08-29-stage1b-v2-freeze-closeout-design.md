# Stage1B v2 冻结、交接、清洗与发布设计

> 状态：用户已于 2026-08-29 批准本设计的聊天版本；本文件用于在实施前固定边界。

## 1. 目标

本次收尾把已经通过确认门禁的 Stage1B 对象统一命名并冻结为 `v2`，不再使用
`v2-r1`、`v2-r2` 或人工递增 revision。冻结对象是：

```text
lorenz96_twoscale_v2
+ ITransformerReference
+ tuned VAR
+ h1–6 主要解释范围
```

完成后，仓库应提供可验证的小型冻结制品、具体的 Stage1B 权威交接快照、经过测试的简化
代码以及干净的本地工作区。GitHub 只接收复现和交接所需的代码、配置、文档与小型结构化
证据，不接收服务器数据库、模型 checkpoint、日志、源码缓存或大型实验包。

## 2. 权威结论与声明边界

确认运行完成 33/33 个任务，失败 0 个。`lorenz96_twoscale_v2` 的主要 h1–6 比较为
72/72 胜，CRPS skill 为 `+13.527369%`，已见和未见状态胜率均为 100%。神经模型的
主要平均绝对校准误差为 `0.011006`，最大值为 `0.030147`，低于 `0.05` 上限；95%
paired bootstrap 改善区间为 `[0.052668, 0.055253]`。

h7–12 仅作为次要描述：61/72 胜、CRPS skill `+1.807016%`。h13–24 是预先声明的
能力边界：25/72 胜、CRPS skill `-1.036598%`。冻结声明不得扩张为“所有时距都优于
VAR”。E01 和 E02 仍未运行。

## 3. 单一 v2 冻结模型

### 3.1 路径与身份

冻结器只接受科学系列 `v2`，标准路径为：

```text
artifacts/stage1b/frozen/v2/qualification_receipt.json
artifacts/stage1b/frozen/v2/manifest.json
artifacts/stage1b/frozen/v2/manifest.sha256
artifacts/stage1b/active.json
```

`active.json` 只记录 `series: v2` 和活动 manifest 的 SHA-256，不记录 revision。manifest
绑定完整 qualification receipt、来源 commits、世界配置、资格配置、执行证据、选定世界、
模型、VAR 基线、主要时距和确认种子。

### 3.2 冻结与覆盖

首次冻结必须满足套件 PASS、来源证据有效、任务完整、无来源/身份漂移、确认 seed 与
E01/E02 保留 seed 分离。已有 `v2` 时，普通调用必须拒绝覆盖。只有带用户身份和原因的
显式授权才能原子替换活动 `v2`；新 manifest 记录前一个 manifest 的 SHA-256。系统不创建
或公开 `r1/r2` 编号。

冻结制品写入后设为只读。验证器重新计算 manifest 和 receipt 哈希，并核对 `active.json`。

## 4. CLI 与兼容性

`scripts/run_stage1b_qualification.py freeze` 删除 `--revision-id` 和
`--prior-revision-id`。保留 `--series v2`；未来覆盖只接受
`--authorize-override --authorization-reason <reason>`。覆盖所需的前一 manifest 身份由程序
从已验证的活动指针读取，不能由调用者伪造。

`scripts/check_stage1b.py` 不再接受 revision 参数。不存在活动冻结时，只有显式
`--allow-unfrozen` 才返回 `UNFROZEN`。

本仓库当前没有 Stage1B 活动指针，因此本次是首次 `v2` 冻结，不制造一个虚假的失败
`v2-r1`。

## 5. Stage1B 权威交接快照

创建 `docs/auth/TARCA_STAGE1B_HANDOFF_SNAPSHOT_2026-08-29.md`，至少包含：

1. 权威层级、阶段目标与范围；
2. v1 失败历史和 v2 首次 pilot 失败证据；
3. 本轮确认配置、服务器环境、任务图、来源、模型和种子；
4. 完整主要、次要和能力边界指标；
5. 冻结制品、代码、配置、来源和实验包哈希；
6. 已实现入口、运行监督、检查点和恢复能力；
7. 当前未完成事项、已知限制和禁止扩张的结论；
8. Stage2、E01、E02、Stage3/4 的标准交接顺序；
9. 本地重现、验证和大型证据恢复方法。

新增一份权威变更控制记录，说明用户取消 Stage1B v2 的 revision 编号。当前活动规范、README
和研究状态文档同步改为 `FROZEN_V2`。历史文档保留当时事实，但明确其 revision 说法已被
新变更控制取代。

## 6. 代码简化边界

逐个扫描所有跟踪的 Python、TypeScript、TSX、JavaScript、shell 和 PowerShell 文件，记录
行数、静态检查和复杂度结果。没有冗余或职责问题的文件不做无意义改写。

必须修改的范围：

- `src/tarca/stage1b/freeze.py`：删除 revision 解析、目录和下一 revision 状态机，拆分 receipt
  验证、manifest 构造、原子发布和验证职责；
- `scripts/run_stage1b_qualification.py`、`scripts/check_stage1b.py`：删除重复 revision 参数与
  校验；
- `tests/stage1b/test_freeze.py`、`tests/stage1b/test_stage1b_cli.py`：以 TDD 固定单一 `v2`
  行为。

全仓热点审计优先处理超过 800 行或复杂度显著超限、且能够在不改变公开接口和科学身份的
前提下拆分的文件。候选包括 `execution/state.py`、`stage1b/runner.py`、`stage1b/jobs.py` 和
`stage1b/training.py`。拆分通过内部模块和兼容性重导出来保持调用方稳定；若某个热点的复杂性
来自必要的契约字段而不是重复职责，则记录理由，不为满足数字而破坏接口。

监控模块属于已交付运行能力，必须补齐 Python 3.11 的可安装 runtime 依赖，使全量测试可以
在隔离 Anaconda 环境中收集和运行。

## 7. 本地清洗策略

所有删除都使用精确路径白名单，删除前验证目标仍位于仓库内并记录文件数与字节数。不得对
工作区根目录、未知路径、其他 worktree 或用户密钥执行递归删除。

### 7.1 保留

- `artifacts/stage1b/server-archives/tarca-stage1b-confirmation-r2-fd27aa4.tar.gz` 及 SHA-256；
- 新的 `artifacts/stage1b/frozen/v2/` 小型结构化证据；
- `stage1b-v2-official-sources.tar.gz` 及来源收据；
- `third_party/stage1b/` 官方固定 commit 缓存，供离线复现与集成测试；
- `docs/research/stage1b_world_qualification_report_v1.md`；
- Stage0/Stage1A 权威制品和用户已有的未跟踪 Stage0 快照。

### 7.2 删除

- 旧 v2 pilot 的大型服务器压缩包和 2.36 GB 解包副本，结果先写入交接快照；
- 当前确认运行的 785 MB 重复解包副本，完整压缩包和冻结 receipt 验证后再删；
- 旧 commit 的代码/前端传输胶囊、bundle 调试目录和 import-audit 工作副本；
- `data/third_party/interfere` 历史 v1 checkout；
- `.venv`、coverage、Python/TypeScript 缓存、构建目录、`node_modules`、测试临时目录和空
  `.tarca-runtime`。

`.gitignore` 改为默认忽略全部 Stage1B 大型制品，只显式允许 `active.json` 和
`frozen/v2` 中规定的 JSON/SHA-256 文件。

## 8. 测试与验证

在仓库外创建新的隔离 Anaconda Python 3.11 环境，不修改现有环境。验证包括：

- 新冻结测试的 RED→GREEN 周期；
- 完整 Python 测试与至少 80% branch coverage；
- Ruff、Mypy 和额外复杂度审计；
- 前端单元测试、coverage、TypeScript 构建和 Playwright 关键流程；
- 冻结 receipt、manifest、活动指针的重载、篡改和授权覆盖测试；
- 最终实验压缩包和冻结文件哈希复核；
- 敏感信息、超大文件、生成物和 Git 暂存清单审计。

无法通过的检查必须修复或明确报告；不得删除测试、降低覆盖率阈值或静默跳过正式检查。

## 9. Git 与 GitHub 发布

当前 `main` 与 `origin/main` 都指向 `8360a9d`，当前功能分支是其后 48 个提交的直接后代。
实施过程中只显式暂存批准的代码、配置、文档和小型冻结证据，不使用会吸入全部未跟踪文件的
宽泛暂存命令。

功能分支验证通过后：

1. 获取远端最新状态；
2. 将本地 `main` 快进到已验证功能分支；
3. 在合并后的 `main` 重新执行完整验证；
4. 检查提交文件、大文件、二进制和秘密；
5. 仅在远端仍可安全快进时推送 `origin/main`；
6. 禁止强推。

其他 worktree 和分支不删除、不移动、不修改。

## 10. 完成条件

- Stage1B 活动身份只有 `v2`，冻结验证为 PASS；
- 权威交接快照能够独立指导下一任务；
- 大型中间副本和可再生缓存已按白名单清理；
- 保留的最终压缩包、来源胶囊和冻结制品哈希有效；
- 全仓审计完成，实际冗余热点已在保持行为的前提下拆分；
- Python、前端、静态和安全检查通过；
- 本地 `main` 与 GitHub `origin/main` 指向同一已验证提交；
- GitHub 不包含 checkpoint、数据库、日志、源码缓存、传输胶囊或大型实验包。
