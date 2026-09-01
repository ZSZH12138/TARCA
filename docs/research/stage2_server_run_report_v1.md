# TARCA Stage 2 v1 服务器运行报告

## 1. 运行结论

2026-09-01，Stage 2 v1 在用户授权的双 RTX 4090 服务器上完成。恢复沿用原运行
`run-acff24d96653a25d4aac54b9389c605d8c35293cc930f9fa8a560947306401fb`，没有重新
`launch`，没有改写科学配置，也没有访问 E02 formal 数据。

最终账本满足：

- 固定图最新状态为 `37/37 COMPLETED`；
- 原六个 `attempt-1 / WORKER_ERROR` 记录完整保留；
- 同一运行新增的六个 `attempt-2` 全部为 `COMPLETED`；
- 六个完整神经检查点按书面恢复契约执行零训练步续接，没有重写检查点；
- Stage 2 freeze receipt 可独立重算，`formal_access_event_count = 0`；
- E02 未准备、未预检、未授权、未运行。

## 2. 服务器接入与兼容性事实

连接继续遵循 `docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md`：User 作用域环境变量优先、
白名单解析连接指令、固定探针先行、临时受限密钥副本、错误输出分类和最终清理。最小检查
确认 SSH 已经落在目标 CUDA 容器内而不是 Docker 宿主机，因此使用
`deploy/stage2/recovery_bootstrap_direct.sh`，没有在容器内安装或嵌套 Docker。

目标容器没有 `python` 命令，但 `/opt/conda/bin/python` 同时满足 Python 3.10、PyTorch
2.2.2 和 pip 契约。直跑入口已改为验证并选择兼容解释器，并用
`venv --system-site-packages --without-pip` 继承镜像 CUDA/PyTorch、避免依赖镜像中的
`ensurepip`，其余依赖仍从包内 wheelhouse 按哈希锁离线安装。

本次服务器最初接收的固定包 SHA-256 为
`c0c0b8da1804e982a26234181d5daa83db0c76cf22bef409f788d78fa2c89dc4`；只替换了经 SHA-256
校验的容器直跑入口，入口 SHA-256 为
`7298cbc50740e25c8c1f34a22a771e1717304f224022ed250949c4c98533b89f`，训练源码、科学配置和
恢复输入未改。修复已重新打入当前本地服务器包，当前包 SHA-256 为
`1d05dd8a98178ef111990131b682552e5b9cd51e1b23f79397bc4a4fec99deee`。

## 3. 冻结身份

- Stage 2 freeze receipt SHA-256：
  `37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166`
- Scientific SHA-256：
  `c2df021d248c2ffcdcf6133179f4b88c86ea88ae4e3f72630f302b88402e0e32`
- 冻结 strongest linear：`VAR`
- 冻结 primary iTransformer seed：`1797287582`

这些是 Stage 2 validation 冻结身份，不是 E02 PASS/FAIL 结论。

## 4. 本地回收与独立校验

完整服务器归档位于：

`artifacts/stage2/server-results/stage2-v1-complete-20260901T011423Z/tarca-stage2-v1-complete-20260901T011423Z.tar.gz`

归档 SHA-256 为：

`7d77ca6bd7d09b3fd9abd7814ef169f11814e774ec479a04d6f1a1c751fe966a`

同目录保存 `.sha256` sidecar 和安全解压副本。归档包含完整 `artifacts/stage2`、运行数据库、
恢复 capsule、冻结 manifest/receipt、已发布 artifact、失败历史和 Stage 2 运行/前端日志。
本地独立校验完成了：外层 SHA-256、归档路径安全、冻结 receipt 重算、最新 attempt 状态、
六个恢复 attempt、旧失败留痕、Stage 2 存储 artifact 内容哈希/完成标记、恢复 capsule 哈希和
E02 不存在性检查。

## 5. 后续边界

Stage 2 已完成，下一科学阶段是 E02，但 E02 仍需要用户单独书面授权。任何 E02 操作前都应
先以本地冻结输出为输入重新执行 E02 的 prepare、dry-run 和 preflight；formal launch 仍必须
使用独立确认串，不能把本次 Stage 2 授权延伸到 E02。
