# Stage1B 离线官方源码包设计

> 状态：用户已授权实施
> 日期：2026-08-27
> 实施分支：`codex/stage1b-runtime-supervision-fix`

## 目标

让服务器在已经关闭、重新开启后，能够使用本地已审核的官方 Git 源码直接运行 Stage1B
预检与资格任务，不再要求服务器访问 GitHub。该调整只改变源码进入服务器的路径；不改变
世界、数据、模型、seed、训练预算、Gate、74 任务图、资源调度、checkpoint 语义或只读监控前端。

## 选择的边界

本地机器是唯一允许从官方 GitHub 拉取源码并审核的地方。每个已锁定来源在本地执行现有的
精确 commit、工作树、关键资产 SHA-256 和完整树 SHA-256 校验后，导出为 Git bundle。所有
bundle 和一份规范化 manifest 再封装为一个源码包，并生成包含外层 SHA-256 的独立 receipt。

服务器只接收这两个文件。导入时它先核对外层包 SHA-256、manifest 哈希、来源集合、仓库 URL、
commit、授权 ID、每个 bundle 哈希，再从本地 bundle 重建干净的 detached checkout。重建后的
checkout 必须再次通过现有 commit、关键资产和树哈希校验，才会原子地发布到源码缓存。服务器
不会把任何 bundle 配置为远端 remote，也不会在离线模式下尝试 GitHub 下载。

## 运行模式

`TARCA_STAGE1B_SOURCE_MODE=offline-capsule` 表示正式服务器运行模式：

1. 缓存中已有的已验证 checkout 可以被读取；
2. 缓存缺失、Git 状态不干净、commit/资产/树哈希不符时立即失败；
3. 不发生网络 fetch、clone 或 remote add。

未设置该变量时仍保持既有本地在线 materializer，方便在本地建立和更新可审核的源码包。
模型适配器、74 任务图中的 `SOURCE_MATERIALIZE` 任务和预检使用同一个缓存根与模式，因此不会
出现预检离线、训练时却又访问 GitHub 的旁路。

## 安全与可恢复性

- 包含 Git 对象而非复制工作目录，避免把 `__pycache__`、本地修改或未追踪文件传入服务器；
- 导入写入临时目录，所有验证通过后才替换对应 `<source_id>/<commit>`，失败时不会破坏已验证缓存；
- manifest 和 receipt 是可审计的 JSON；错误会说明是外层哈希、manifest、bundle、身份还是源码树失败；
- 正式运行仍使用现有 SQLite、制品仓库和 checkpoint；服务器断电后 `resume` 逻辑不变；
- 调度器的双卡单任务准入、CPU affinity 合同、真实 NVML/进程采样和前端 API 均不修改。

## 用户流程

```text
本地：下载并审计六个官方来源
→ 生成 source-capsule.tar.gz + source-capsule.receipt.json
→ 通过服务器安全通道上传两个文件
→ 服务器：离线导入并逐项验证
→ offline-capsule 预检
→ launch / 中断后 resume
→ 原有只读前端监控真实调度与资源
```

## 验收标准

1. 篡改源码包、manifest 或 bundle 会在导入前失败，且不覆盖已有缓存；
2. 离线缓存缺失时 materializer 明确失败，并且测试证明未调用网络 Git 命令；
3. 导入后每个来源与本地审核时的 commit、关键资产和树哈希一致；
4. 运行器、任务执行器和两个模型适配器都尊重同一离线缓存根；
5. 现有调度、checkpoint、监控 API 和前端回归测试继续通过；
6. 不执行完整 Stage1B、E01、E02 或冻结。
