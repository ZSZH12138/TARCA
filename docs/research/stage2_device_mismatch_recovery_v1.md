# TARCA Stage 2 设备边界事故恢复规格 v1

日期：2026-08-31

> 后续状态（2026-09-01）：本文只保留 Stage 2 设备边界事故的恢复规格。该恢复已经完成，
> 随后的 E02 已完成 `COMPLETED / PASS`；当前正式结论与后续边界见
> `docs/auth/TARCA_E02_HANDOFF_SNAPSHOT_2026-09-01.md`。

## 1. 恢复目标

本规格只处理 run
`run-acff24d96653a25d4aac54b9389c605d8c35293cc930f9fa8a560947306401fb`
中六个 `NEURAL_TRAIN` 任务的设备边界事故。机器可执行的唯一权威范围位于
`configs/stage2/stage2_device_mismatch_recovery_v1.json`；本文解释其功能含义，不扩大范围。

恢复不创建新实验、不更换科学配置、不删除失败记录、不重训已完成模型，也不接触 E02 formal
数据。目标是复用六个已经密封为 `COMPLETE` 的 checkpoint，在同一 run 内继续完成 Stage 2。

## 2. 已确认的故障

训练和 checkpoint 写入已经完成，但训练任务结束前的固定验证预测把 CPU 张量交给了 CUDA
模型。PyTorch 因模型和输入不在同一设备而抛出异常；通用 worker 把异常记为
`WORKER_ERROR`，六个 attempt-1 因而显示为失败。

这意味着“训练结果不存在”不是事实。六个 checkpoint 都能加载、身份匹配、状态为
`COMPLETE`，并且 SHA-256 已冻结。需要重做的是训练任务最后的设备一致预测和后续图节点，
而不是 0 到 100 epoch 的训练。

## 3. 不可变恢复规则

- 只接受固定恢复归档
  `tarca-stage2-recovery-20260831T102151Z.tar.gz`，SHA-256 为
  `79c6cb2c0f8fd8a1801d378fb779212b66f3774d8372df7b360b1721b3f9b126`。
- 只恢复 JSON 规格列出的 PatchTST/iTransformer × 3 seeds，共六个任务。
- 六个 attempt-1 永久保留为 `FAILED / WORKER_ERROR`，用于事故审计。
- 每个任务只允许追加 attempt-2；重复执行 repair 返回同一凭证，不产生 attempt-3。
- attempt-2 必须加载指定 `COMPLETE` checkpoint，训练循环执行 0 步，checkpoint 文件不得重写。
- checkpoint 缺失、哈希变化、身份变化、状态不是 `COMPLETE`、数据库变化或代码包不匹配时，
  必须在执行任务前停止。
- 模型固定验证、checkpoint 验证都把输入移动到模型实际所在 GPU，并使用 eval/inference 模式。
- 任何恢复动作均不修改 Stage 2 scientific hash
  `8a0509edfd1487dc36188e8d12ca088d52f0287804f4808215ff0f7c279c069f`。

## 4. 资源和并行语义

恢复预检同时在 GPU 0 和 GPU 1 上各加载一个完整 checkpoint；正式 resume 保持两个独立的
单卡 worker，调度器只要存在两个可运行 GPU 节点，就同时占用两张卡。每个 `NEURAL_TRAIN`
节点仍按冻结图申请 4 CPU cores、20 GiB VRAM 和 32 GiB RAM；`CHECKPOINT_VALIDATE` 申请
2 CPU cores 和 8 GiB VRAM。

主机准入预算仍是 24 个工作核、1 个调度/监控核、3 个系统/I/O 核。这里的“吃满硬件”指不让
可运行 GPU 工作串行化，并允许满足依赖的 CPU 节点 backfill；不能虚构额外任务或改变冻结的
资源请求来制造 100% CPU/GPU 曲线。后续 `VALIDATION_PREDICT` 在原图中是 CPU 节点，继续按
原图执行。

## 5. 恢复流程

1. 校验固定服务器包和恢复归档的外层 SHA-256。
2. 安全导入归档：只提取 Stage 2 artifacts、Stage1B v2 和 E01 v2 冻结输入；归档中的旧源码
   和旧热修复文件一律忽略。
3. 用一致性 SQLite 快照作为当前执行数据库，签发与新服务器包绑定的恢复输入凭证。
4. 运行最小服务器检查，并在两张 4090 上并发执行两个完整 checkpoint 的只读预测；训练步数
   必须为 0，checkpoint 哈希前后必须一致。
5. repair 在同一 run 内追加六个 READY attempt-2，写入六条 `recovery_events` 和只读监控告警。
6. 先启动只读前端，再由用户单独确认 resume。前端只展示每个任务的最新 attempt，因此旧失败
   不再冒充当前状态；事故记录仍保留在账本中。
7. 两张 GPU 并发完成六个快速恢复节点和六个 GPU checkpoint 验证，随后按原图完成 validation、
   selection、freeze 和 receipt。

服务器一键准备入口为 `deploy/stage2/recovery_bootstrap.sh`。该脚本执行到监控可用和恢复任务
READY 为止，不会自行执行 resume；它最后打印需要用户再次确认后才运行的命令。

## 6. 完成判据

- 恢复输入凭证、服务器预检凭证和恢复授权凭证均能独立校验；
- 六个 checkpoint 的 SHA-256 与固定规格完全一致，恢复期间未改写；
- 两张 GPU 在存在两个 ready GPU 节点时并发运行，前端能看到 GPU 0/GPU 1、真实遥测、CPU
  亲和性、任务状态和 ETA 来源；
- 同一 run 的 37 个图节点全部完成并发布 Stage 2 freeze receipt；
- E02 仍未打开，除非用户之后给出第二次独立授权。
