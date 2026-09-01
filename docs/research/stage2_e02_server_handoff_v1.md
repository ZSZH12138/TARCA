# TARCA Stage 2 / E02 服务器交接 v1

> 2026-09-01 状态更新：本文件描述的同运行恢复已经完成，Stage 2 为 `37/37 COMPLETED`，
> freeze receipt 与本地完整归档均已独立校验，E02 仍未打开。当前事实与回收位置见
> `docs/research/stage2_server_run_report_v1.md`；下文恢复步骤保留为事故审计记录，不再是待执行任务。

## 1. 当前边界

上一台服务器已经执行过 Stage 2 的前半段：37 个图节点中形成了 16 个完成记录，六个神经
训练 attempt-1 在训练和 `COMPLETE` checkpoint 写入之后，因固定验证预测的 CPU/CUDA 设备
不一致而失败。完整服务器状态已经下载为固定恢复归档；当前本地状态为
`LOCAL_RECOVERY_KIT_READY`，E02 尚未打开。

下一台服务器不得重新 `launch` 或从头训练。唯一允许的 Stage 2 路径是：导入固定恢复归档、
运行恢复专用 preflight、在同一 run 中追加六个 attempt-2、启动只读前端，然后等待用户再次
确认 `resume`。本文件中的确认串只是精确 CLI 契约，不构成当前对远程连接、上传、resume 或
E02 formal open 的授权。

服务器访问必须遵守 `docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md`：只从受控环境变量读取连接
事实，不打印凭据，不执行环境变量中的原始命令；先运行固定 `TARCA_SERVER_PROBE_OK` 探针，
再执行用户当前明确授权的白名单操作，并在 `finally` 清理临时密钥、known_hosts、代理脚本
和残留进程。

## 2. 恢复包与开机入口

通过既有安全通道把下面五个文件上传到新服务器同一目录：

```text
tarca-stage2-v1-server.tar.gz
tarca-stage2-v1-server.tar.gz.sha256
tarca-stage2-v1-server.tar.gz.receipt.json
tarca-stage2-recovery-20260831T102151Z.tar.gz
tarca-stage2-recovery-20260831T102151Z.tar.gz.sha256
```

服务器目录应位于至少有 300 GiB 可用空间的本地 NVMe。先校验并解压固定代码包：

```bash
sha256sum --check tarca-stage2-v1-server.tar.gz.sha256
sha256sum --check tarca-stage2-recovery-20260831T102151Z.tar.gz.sha256
mkdir -p /opt/tarca
tar -xzf tarca-stage2-v1-server.tar.gz -C /opt/tarca
cd /opt/tarca
```

服务器包已经携带本地测试和构建完成的只读前端，不在服务器执行 `npm ci`，也不需要 Node
构建前端。正式论文代码来源从包内 capsule 导入，不从网络追踪 GitHub 分支。如果 SSH 目标是
Docker 宿主机，使用下述 Compose 恢复入口；如果云平台已经把 SSH 会话放在用户指定的
`pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04` 容器内且没有 Docker daemon，则使用包内
`recovery_bootstrap_direct.sh`，不得在容器内安装 Docker。

`N` 是此刻距整机清空的真实剩余小时数。运行恢复准备脚本：

```bash
bash deploy/stage2/recovery_bootstrap.sh \
  --recovery-archive /path/to/tarca-stage2-recovery-20260831T102151Z.tar.gz \
  --server-bundle /path/to/tarca-stage2-v1-server.tar.gz \
  --remaining-rental-hours N
```

已处于目标 PyTorch 容器内时，等价的直接入口为：

```bash
bash deploy/stage2/recovery_bootstrap_direct.sh \
  --repository-root /opt/tarca \
  --recovery-archive /path/to/tarca-stage2-recovery-20260831T102151Z.tar.gz \
  --server-bundle /path/to/tarca-stage2-v1-server.tar.gz \
  --remaining-rental-hours N
```

直接入口只使用包内 wheelhouse 创建继承系统 PyTorch/CUDA 的隔离虚拟环境，然后执行相同的
来源导入、恢复、双卡 preflight、repair 和只读前端启动；它同样停在
`RECOVERY_READY_FOR_USER_RESUME`，不会自行 resume。

该脚本 fail closed 地按以下顺序执行：

1. 再次校验两个外层 SHA-256，并构建固定镜像；
2. 只从恢复归档提取 Stage 2 artifacts、Stage1B v2 和 E01 v2 冻结输入，忽略归档里的旧源码；
3. 用一致性快照恢复原执行数据库，并把恢复输入凭证绑定到当前服务器包；
4. 做最小硬件/来源检查，同时在 GPU 0 和 GPU 1 上各运行一个完整 checkpoint 的只读预测；
5. 要求训练步数为 0、checkpoint 哈希不变，并验证 `ETA + 1 hour < remaining rental window`；
6. 在同一 run 中追加六个 READY attempt-2，保留 attempt-1 失败记录；
7. 启动只读监控，等待 `/api/v1/run` 可用；
8. 输出 `RECOVERY_READY_FOR_USER_RESUME` 和下一条命令，但不自动 resume。

恢复专用 preflight 不再重复最大 epoch 训练；它只验证真正要走的 `COMPLETE` checkpoint 快速
路径，并用两卡并发观测、三波 GPU 工作、35% 裕量和固定非神经开销给出保守 ETA。任一
checkpoint 缺失、哈希/身份漂移、GPU 不可用、来源漂移、租期不足或非有限预测都会停止。

## 3. 前端监督

恢复准备脚本成功后，监控容器已经启动。服务只绑定 `127.0.0.1:8765`。通过符合服务器
runbook 的 SSH 隧道访问 `http://127.0.0.1:8765/`；不要把 8765 暴露到公网。

页面每 2 秒刷新主机、GPU 0、GPU 1、真实利用率、显存、功率、温度、计算进程、CPU 忙核和
亲和性。尚未分配的 GPU 任务明确显示“GPU 待分配”和冻结的显存请求，不会误标为 CPU 任务。
任务表只展示每个任务的最新 attempt，因此 repair 后显示 READY/RUNNING attempt-2，
不会把保留的 attempt-1 事故记录误报为当前失败。整体 ETA 在运行历史形成前使用哈希绑定的
服务器预检保守估计，并明确标记来源；遥测缺失显示“不可用”，不会伪装成 0。

监控 API 只允许读取，不提供停止、重试、删除或修改任务的按钮。用户和前端承担运行监督；
不需要 Codex 持续轮询服务器。

## 4. 用户确认后的 Stage 2 resume

看到 `RECOVERY_READY_FOR_USER_RESUME`、前端可用和 preflight 证据之后，用户仍需单独确认
resume。当前数据库已经存在，禁止使用 `launch`。精确确认串为
`I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN`：

```bash
docker compose -f deploy/stage2/compose.yaml run -d \
  --name tarca-stage2-recovery-resume tarca-stage2 stage2 resume \
  --repository-root /opt/tarca \
  --config configs/stage2/stage2_v1.yaml \
  --artifact-root artifacts/stage2 \
  --acknowledgement I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN
```

调度器优先填充两个 GPU 槽：只要至少有两个可运行的 GPU 节点，就分别在 GPU 0/GPU 1 上
并行启动，绝不把六个恢复任务串行化。每个任务仍使用冻结资源请求，CPU-only 尾部任务在依赖
满足时 backfill；不通过伪造额外任务制造虚假占用。

辅助状态查询不会打开科学结果：

```bash
docker compose -f deploy/stage2/compose.yaml run --rm tarca-stage2 stage2 status
docker logs --tail 200 tarca-stage2-recovery-resume
```

进程中断时保留 `artifacts/stage2/`，使用相同命令和同一科学图 resume；如果命名容器仍存在，
先检查其真实状态，不得用新 launch 隐藏旧失败。

## 5. 24 小时清空与恢复

`TARCA_24H_RESET` 表示第 24 小时整机断电并清空；Docker 命名卷、宿主机 bind mount 和容器
层都不能被当作跨机器持久化。`TARCA_RESET_MARGIN` 固定为 1 小时：点火时剩余租期必须严格
大于实测 ETA 加 1 小时，否则等待一台新的完整租期服务器。

任何异常或接近 reset margin 时，先生成内容寻址恢复 capsule：

```bash
docker compose -f deploy/stage2/compose.yaml run --rm tarca-stage2 stage2 recover
```

随后把 `artifacts/stage2/runtime/recovery/` 中最新目录、执行数据库和全部已发布 artifact 下载到
本地或持久对象存储，并核对 recovery receipt。新机器上必须重新生成一个与新快照精确绑定的
恢复规格和恢复包，不能把本次六任务设备事故的固定 repair 规格套到另一状态；不得用新 launch
隐藏旧失败。

## 6. E02 是第二次独立授权

只有 Stage 2 已产生并通过核验的
`artifacts/stage2/frozen/v1/stage2_freeze_receipt.json` 后，才可执行不打开 formal 数据的准备：

```bash
docker compose -f deploy/stage2/compose.yaml run --rm tarca-stage2 e02 prepare
docker compose -f deploy/stage2/compose.yaml run --rm tarca-stage2 e02 dry-run
docker compose -f deploy/stage2/compose.yaml run --rm tarca-stage2 e02 preflight
```

重新检查剩余租期和资源；必要时应在新服务器完整租期开始后再做 E02。必须由用户单独明确
授权 E02 后，才可使用确认串 `I_ACKNOWLEDGE_E02_V1_FORMAL_RUN`：

```bash
docker compose -f deploy/stage2/compose.yaml run -d \
  --name tarca-e02-run tarca-stage2 e02 launch \
  --acknowledgement I_ACKNOWLEDGE_E02_V1_FORMAL_RUN
```

E02 launch 才会原子创建 sealed access grant 并首次物化 120 条 formal trajectories。中断后只
能按相同 run identity 执行：

```bash
docker compose -f deploy/stage2/compose.yaml run -d \
  --name tarca-e02-resume tarca-stage2 e02 resume \
  --acknowledgement I_ACKNOWLEDGE_E02_V1_FORMAL_RUN
docker compose -f deploy/stage2/compose.yaml run --rm tarca-stage2 e02 recover
```

E02 运行时若需要同一监控端口，应停止 Stage 2 monitor，再以 E02 数据库启动监控容器：

```bash
docker compose -f deploy/stage2/compose.yaml stop tarca-stage2
docker compose -f deploy/stage2/compose.yaml run -d --service-ports \
  -e TARCA_RUNTIME_DATABASE=/opt/tarca/artifacts/e02/runtime/execution.sqlite3 \
  -e TARCA_EXECUTION_KIND=e02-v1 --name tarca-e02-monitor tarca-stage2 monitor
```

## 7. 操作完成的判据

- 上传完成：外层 SHA-256 与 receipt 一致；
- 环境可用：bootstrap 输出 `PREFLIGHT_PASS` 且 evidence 中 formal task count 为 0；
- Stage 2 完成：固定图全部任务完成、freeze receipt 可独立校验；
- E02 完成：120/120 trajectory、全部预测/评分/bootstrap/guardrail 完成并原子发布 E02 receipt；
- 任何时候都不能把“容器启动”“GPU 利用率高”或“中间分数看起来好”替代科学完成凭证。
