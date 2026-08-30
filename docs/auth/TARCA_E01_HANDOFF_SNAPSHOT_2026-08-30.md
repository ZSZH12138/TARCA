# TARCA E01 完成情况与任务交接快照

> 快照日期：2026-08-30
> 阶段状态：`COMPLETED / PASS`
> 唯一活动版本：`v2`
> 实验身份：`e01_scm_truth_v2`
> 前置身份：Stage1B `FROZEN_V2`
> 下一阶段：Stage 2 基础预测器与概率输出，随后实施 E02
> 变更控制：`TARCA_PROTOCOL_CHANGE_CONTROL_CCP_0004.md`

## 1. 功能层结论

E01 已完成。它没有训练 Stage 2 模型，而是检查 TARCA 后续实验使用的“效果测量尺”是否可信。
最终结果说明：当正确答案预先已知时，测量程序能够稳定测到正确大小、正确延迟和足够窄的区间；
面对错误 SCM、错误 lag 和随机概念时，也能把错误答案区分出来；在没有干预时不会制造假效果。

因此，E01 对项目给出的结论是：

```text
SCM / paired-effect oracle 可以作为 Stage 2 之后机制实验的可信测量基础。
```

这只解除“测量尺不稳定”的阻塞，不代表预测模型已经完成，也不授权直接进入 Stage 3/4。

## 2. E01 与 Stage1B、Stage 2 的关系

Stage1B 先冻结了一个有生成器真值、干预结构和神经内部位置的正式世界：
`lorenz96_twoscale_v2 + ITransformerReference + h1–6`。E01 随后验证测量这个世界和解析真值时所用
的 effect oracle。两者共同构成 Stage 2 及后续机制研究的上游基础：

```text
Stage1B：固定“在哪里研究、研究什么模型”
E01：确认“用什么尺子测干预效果”
Stage 2：训练并冻结概率预测器输出
E02：正式验收 Stage 2 的 NLL / CRPS
Stage 3/4：只有上述前置都满足后才进入机制植入与交换干预
```

## 3. 版本沿革与合并规则

### 3.1 v1 历史

E01-v1 完成 166/166 个任务，运行失败任务 0、告警 0，但整体科学 Gate 为 `FAIL`：

- 解析 E01-A 的旧端点收敛规则只通过 2/5 个种子，低于 4/5 要求；
- Lorenz-96 E01-B 的收敛为 5/5，三类方向性对照也各为 5/5，属于有效成功证据。

v1 失败原因是“少量种子的单次端点误差必须近似单调减半”容易因 Monte Carlo 抽样波动误拒正确
方法，不是服务器任务失败，也不是 Lorenz-96 失败。CCP-0004 在正式 v2 结果产生前冻结了校准后
的 50 种子规则。

### 3.2 v2 最终组成

最终活动 `v2` 合并两部分：

1. v2 正式重新运行的解析 E01-A；
2. v1 中已通过、经原字节和 SHA-256 独立验证的 Lorenz-96 E01-B。

v1 的失败部分没有进入 v2 成功判定。v1 的最终报告和回收验证按原字节 Base64 封装在唯一历史
文件中，各自原始 SHA-256 保持不变。仓库不再保留第二套 v1 配置、运行器、部署入口或失败数据。

配置和正式报告中仍出现 `analytic_delayed_control_v1`，这是正式运行时已经冻结的“解析世界模型
标识”，不能在结果产生后改名；它不代表 E01-v1 仍是活动版本。活动实验身份只认
`e01_scm_truth_v2` 和 `artifacts/e01/active.json`。

## 4. E01-v2 正式结果

服务器任务图包含 50 个 GPU 生成任务、50 个 CPU 分析任务和 1 个汇总任务，共 101 个：

| 项目 | 结果 |
|---|---:|
| 计划任务 | 101 |
| 完成任务 | 101 |
| 失败任务 | 0 |
| 运行警报 | 0 |
| E01-A | PASS |
| E01-B | PASS |
| 整体 Gate | PASS |

### 4.1 E01-A 检查

| 检查项 | 实际结果 | 通过要求 | 判定 |
|---|---:|---:|---:|
| 95% 区间覆盖精确真值 | 48/50 | 至少 45/50 | PASS |
| MCSE 收缩比达标 | 50/50 | 至少 45/50 | PASS |
| 最终区间半宽达标 | 50/50 | 至少 45/50 | PASS |
| 正确 lag 恢复 | 50/50 | 至少 45/50 | PASS |
| identity 逐比特为零 | 50/50 | 至少 45/50 | PASS |
| WRONG_SCM 方向性对照 | 50/50 | 至少 45/50 | PASS |
| WRONG_LAG 方向性对照 | 50/50 | 至少 45/50 | PASS |
| RANDOM_CONCEPT 方向性对照 | 50/50 | 至少 45/50 | PASS |
| 总体乘数偏差 | 0.0001294253 | 不高于 0.005 | PASS |

0.0001294253 约等于 0.01294%，明显低于允许的 0.5%。结果既说明测量值准确，也说明程序没有
对所有输入一律给出“正确”。

### 4.2 E01-B 检查

Lorenz-96 E01-B 使用五个正式种子的成功证据：收敛 5/5，WRONG_SCM、WRONG_LAG、
RANDOM_CONCEPT 三类对照各 5/5，运行告警 0。原最终报告、回收验证、原服务器档案、科学配置和
Stage1B manifest 的哈希全部匹配后，E01-B 才能并入 v2。

## 5. 冻结身份与哈希

| 对象 | SHA-256 / 身份 |
|---|---|
| v2 科学配置身份 | `07c9009f1cd70667bea66e41c591f662012504f7c50eae2c39a1b2610d6ae44e` |
| v2 runtime 配置身份 | `8594c8294dadbb7f7b1d404acc297c8b113d0c105f9b2c63833e07ffe845a681` |
| v2 资格收据内部哈希 | `16de7fc103b8f1589eec07deaebfb66fbf7ea603046020e4778bb52458c3ae14` |
| v2 资格收据文件哈希 | `ed9ba52be11aef8d78f47231659ec58f0f216b56022b8db4cad089b000cee965` |
| v1 历史记录内部哈希 | `a507e3c583ad47f451ed09d86d56ba2c7233985657dd35508aaa1d9e8e1b961b` |
| v1 历史记录文件哈希 | `06f9b597726198b0165e87e51e0d0fe946eb24c18e761f09da0edc9b8a2c36f6` |
| v1 原最终报告哈希 | `c31f04988d09efa7de1cd67d65f604d444f70fcf7c066206b981f38202d61528` |
| v1 原回收验证哈希 | `705e82264a880b79e73ae4e89ef3d531af333912bf4cf0dc967974089df6cd9f` |
| v1 原服务器档案哈希 | `d6d3b4716b31437a8dc437260c8a9e9d422c84474bd3ea9ea2d47aace0de82c4` |
| v2 回收验证文件哈希 | `3df6b12bf417a06d062a522e36c4ba9929562ff23e2d0b59c1324b8d58047e1a` |
| v2 原始结果档案哈希 | `8b49ea6fafe440876977b7c4a55de11dea39e70ba631ae86430081d286ec855d` |
| v2 最终报告内容哈希 | `b8f48acfa7ac1fa2bf9790119b1c3a4458035e16f89c058592ef544ea4eaef29` |
| 正式上传服务器包哈希 | `2d17429c297768f6ef605c2428a904b610ec0044aee777a3d5243bac103881c3` |
| Stage1B manifest | `d1b4d09260bcc41b3b94a020474ee0b5e9f9dd5f0f498bb96510228141f44b25` |

配置中的两个 v1 证据路径已同时指向唯一历史记录；路径字段不属于科学身份，因此整理前后的
v2 科学配置身份仍为 `07c900...ae44e`，没有事后改变科学参数或 Gate。

## 6. GitHub 与本地证据边界

GitHub 只上传代码、测试、配置、文档和三份小型冻结证据：

```text
artifacts/e01/active.json
artifacts/e01/frozen/v2/qualification_receipt.json
artifacts/e01/history/e01_v1_history_record.json
```

以下文件保存在本机但被 `.gitignore` 排除，不上传 GitHub：

```text
artifacts/e01/bundle/tarca-e01-server-v2.tar.gz
artifacts/e01/bundle/tarca-e01-server-v2.tar.gz.sha256
artifacts/e01/bundle/tarca-e01-server-v2.receipt.json
artifacts/e01/recovery/formal-v2-20260830T085824Z/e01-v2-artifacts.tar.gz
artifacts/e01/recovery/formal-v2-20260830T085824Z/e01-v2-artifacts.tar.gz.sha256
artifacts/e01/recovery/formal-v2-20260830T085824Z/e01-v2-artifacts.manifest.json
artifacts/e01/recovery/formal-v2-20260830T085824Z/e01-v2-recovery-summary.json
artifacts/e01/recovery/formal-v2-20260830T085824Z/recovery_validation.json
```

v2 解压副本已删除，因为原始压缩包、校验和、清单和回收验证足以恢复。v1 的失败运行目录、旧
服务器包、热修包和三份分散 carry-forward 文件已移入 Windows 回收站，共释放项目目录约
3.8 GiB；它们在回收站清空前仍可恢复。v1 在活动项目中只剩唯一历史记录。

## 7. 当前代码与操作入口

```text
configs/e01/e01_v2.yaml                    唯一活动科学与服务器配置
src/tarca/e01/v2_*.py                      v2 配置、证据、指标、任务图和运行时
src/tarca/e01/{config,estimators,...}.py   v2 实际使用的最小公共能力
scripts/run_e01_v2.py                      prepare/dry-run/preflight/launch/resume/status
scripts/prepare_e01_v2_server_bundle.py    确定性服务器包生成与封条
deploy/e01/*v2*                            v2 容器、Compose、bootstrap 和 supervisor
tests/e01/test_v2_*.py                     v2 自动化验证
```

v1 的 `run_e01.py`、配置、任务图、Lorenz-96 重跑器、Docker 入口、监控 sidecar 和对应测试均已
删除。`tarca.execution.worker_entry` 也明确拒绝旧 `TARCA_EXECUTION_KIND=e01`，只接受
`e01-v2` 或既有 `stage1b`。

## 8. 服务器环境和恢复规则

本次正式环境：

```text
镜像：pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04
GPU：1 × NVIDIA RTX 4090 24GB
CPU：14 个物理核
RAM：112 GiB
```

服务器已不再需要保持开启。未来只有在证据损坏或协议授权重放时才重新使用服务器；仍须遵守
`TARCA_SERVER_ACCESS_RUNBOOK.md`，重新点火必须再次取得独立授权，不能把本快照当作授权口令。
中断恢复只能使用 `resume` 和同一科学身份，禁止混合新配置与旧 SQLite/ArtifactStore。

## 9. 已知边界

- E01 证明的是合成 SCM 与 Lorenz-96 环境中的 effect oracle 稳定，不证明真实金融世界因果；
- E01-B 是哈希验证后的成功证据并入，不是 v2 再次计算 Lorenz-96；
- v1 的 2.5GB 原始 effect store 已清理，只保留原档案哈希和用于判定的原字节报告/回收验证；
- E01 不评价概率预测器的 NLL/CRPS，因此不能代替 E02；
- E01 通过不允许提前实施内部交换、OT、DAS、DRO 或 zero-refit 主张；
- 大型本地档案不进入 GitHub，后续备份应由本机存储策略负责。

## 10. 下一任务的正确起点

1. 读取本快照、Stage1B 快照、CCP-0004、E01 active 指针和 v2 资格收据；
2. 确认 Stage1B 活动 manifest 仍为
   `d1b4d09260bcc41b3b94a020474ee0b5e9f9dd5f0f498bb96510228141f44b25`；
3. 进入 Stage 2，依次实现 Last-value / Seasonal naive、AR/VAR、DLinear、小型 PatchTST 和小型
   iTransformer；
4. 所有模型统一输出协议定义的 `ForecastDistribution`，训练、验证、测试切分和 normalizer 继续
   使用冻结契约；
5. 至少一个神经概率预测器达到预注册预测门槛后，冻结 checkpoint、split、normalizer、seed 和
   输出契约；
6. 再实施 E02，使用 NLL、CRPS 和校准指标验收 Stage 2；
7. Stage 2/E02 未完成前，不进入 Stage 3/4。

后续任务若只需要判断 E01 是否通过，优先读取
`artifacts/e01/frozen/v2/qualification_receipt.json`；若需要理解历史、服务器、清理边界和下一步，
以本快照为权威交接入口。
