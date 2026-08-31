# TARCA Stage 2 / E02 本地实施报告 v1

日期：2026-08-31

## 1. 结论与状态边界

本地实现状态为 `LOCAL_IMPLEMENTATION_COMPLETE`。Stage 2 和 E02 的科学配置、执行图、正式访问
边界、双 GPU 调度、恢复、只读监控、CUDA 容器和确定性服务器包均已实现。

本报告明确记录：`NOT_RUN_FULL_STAGE2_E02`、`REMOTE_SERVER_NOT_CONNECTED`。尚未执行完整
Stage 2/E02，未连接用户提供的服务器，未生成 E02 formal grant，也不声明 GPU 正式验收、
Stage 2 模型选择结果或 E02 PASS/FAIL 结果。

## 2. 已冻结身份

| 身份 | SHA-256 / 值 |
| --- | --- |
| Stage 2 scientific config | `8a0509edfd1487dc36188e8d12ca088d52f0287804f4808215ff0f7c279c069f` |
| E02 scientific config | `9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c` |
| Stage 2 source capsule | `24ed91eea0789554b4b9417d1fdae367084aecac978a709686f6a0e3302e47dc` |
| Stage1B frozen manifest | `d1b4d09260bcc41b3b94a020474ee0b5e9f9dd5f0f498bb96510228141f44b25` |
| E01 frozen receipt identity | `16de7fc103b8f1589eec07deaebfb66fbf7ea603046020e4778bb52458c3ae14` |

固定官方来源为：

| source | commit |
| --- | --- |
| DLinear | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` |
| iTransformer | `4e938a1767106324dd753b2a44832bf870a0252e` |
| PatchTST | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` |
| Lorenz-96 scoring-rules source | `6f28942f6a703c2b52501d01258ca2708539f209` |

来源 capsule 内含四个 Git bundle、精确 commit、树哈希和关键资产哈希。服务器构建只导入该
capsule，不在正式流程中从 GitHub 追踪分支或下载“最新”源码。

## 3. 已实施内容

- Stage 2：冻结上游核验、三组开发数据、Last Value、Seasonal Naive、VAR、DLinear、
  PatchTST、iTransformer、六个神经初始化训练、checkpoint 验证、validation 预测、固定选择、
  manifest、freeze receipt 与资格 receipt。
- E02：Stage 2 冻结核验、独立 formal grant、120 条 formal trajectories、固定 strongest-linear
  与三个 iTransformer 初始化、逐 trajectory 评分、5000 次分层配对 block bootstrap、guardrail、
  决策和原子 final receipt。
- Execution plane：28 核主机中固定 24 个工作核，1 核调度/监控，3 核系统/I/O；两个 20 GiB
  GPU 任务分别占用 GPU 0/1，CPU backfill 使用剩余预算，科学身份与 worker placement 解耦。
- 服务器 preflight：验证 Python/PyTorch/CUDA、双 RTX 4090、RAM、磁盘、来源、FP32、AMP、
  原子 checkpoint；同时在两张卡上并行执行精确 PatchTST/iTransformer 的 forward/backward/
  checkpoint/reload 探针，以最大 epoch 工作量、35% 测量裕量和 4 小时非神经开销估计 critical
  path。只有 `ETA + 1 hour < remaining rental window` 时才签发 preflight receipt。
- 生命周期：prepare、dry-run、preflight、launch、resume、status、recover；Stage 2 与 E02 使用
  不同的确认串和独立数据库，E02 只能读取已经冻结并校验的 Stage 2 suite。
- 只读监控：仅暴露 GET/HEAD，缺失遥测不会显示为 0，不展示未密封科学中间结果。

## 4. 本地环境和验证证据

本地解释器为 Python 3.11.15；PyTorch 为 `2.13.0+cpu`，`torch.version.cuda` 为空，CUDA
不可用。主机为 6 个物理核、12 个逻辑处理器、约 16.9 GB RAM。因此本地只适合契约、CPU
单元、确定性和 fake-resource 验证，不具备完整训练或 formal E02 的硬件条件。

截至交接前的聚焦验证结果：

- Stage 2/E02 聚焦覆盖测试：115 passed，branch coverage `80.55%`；
- Docker Compose 静态配置：PASS；
- 三个 Bash 脚本语法：PASS，WSL 额外输出一次本地主机代理解析警告，不影响 `bash -n` 退出码；
- Ruff、mypy 和 `git diff --check`：PASS；
- 双 4090 CUDA/吞吐/ETA probe：`NOT_RUN_LOCAL_NO_CUDA`；
- 完整 Stage 2 training：`NOT_RUN_FULL_STAGE2_E02`；
- E02 formal open / scoring / decision：`NOT_RUN_FULL_STAGE2_E02`。

最终全量 pytest、最终 bundle 双构建哈希一致性和秘密扫描将在本地冻结交接前再次执行；最终
服务器包的权威 SHA-256 位于包旁的 `.sha256`，结构化信息位于 `.receipt.json`。哈希不嵌入
包内本报告，以避免自引用改变包本身。

## 5. 服务器适配判断

用户提供的镜像 `pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04` 和硬件
`2 × RTX 4090 24GB、28 CPU cores、224 GiB RAM` 满足冻结的目标 profile。显存准入按
驱动实际可报告值允许每张卡不少于 23 GiB，但每个科学 GPU task 的硬上限仍为 20 GiB。

| 资源 | 最低准入 | 推荐 |
| --- | --- | --- |
| CPU | 28 个可用物理/逻辑核 | 28 核，24 核工作预算 |
| RAM | 224 GiB | 224 GiB，200 GiB 调度 ceiling |
| GPU | 2 × RTX 4090，每张实际可见显存不少于 23 GiB | 2 × RTX 4090 24GB |
| 本地存储 | 200 GiB free | 300 GiB free NVMe |
| 容器共享内存 | 16 GiB | 16 GiB |

硬件“规格足够”不等于无条件允许点火。目标机仍须通过固定 SSH 探针、容器 preflight、两卡
并发代表性吞吐和剩余租期 gate；任一失败均停止，不减少 seed、epoch、trajectory 或模型。

## 6. 交付物

- `configs/stage2/stage2_v1.yaml` 与 `configs/e02/e02_v1.yaml`；
- `src/tarca/stage2/`、`src/tarca/e02/`、共享 execution/monitoring runtime；
- `deploy/stage2/` 的镜像、Compose、bootstrap、entrypoint 和 supervisor；
- `scripts/run_stage2_v1.py`、`scripts/run_e02_v1.py` 与确定性 bundle builder；
- `artifacts/stage2/server-bundles/tarca-stage2-v1-server.tar.gz` 及外部 hash/receipt；
- `docs/research/stage2_e02_server_handoff_v1.md`。

下一步不是修改科学规则，而是在用户另行明确授权后，按服务器交接文档完成固定连接探针、
上传、bundle 校验和 preflight。只有用户随后单独确认 Stage 2 点火，才可运行正式训练；E02
还必须等待 Stage 2 成功冻结并获得第二次独立授权。
