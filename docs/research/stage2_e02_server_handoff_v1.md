# TARCA Stage 2 / E02 服务器交接 v1

## 1. 当前边界

本地代码和离线服务器包已经准备，但 `REMOTE_SERVER_NOT_CONNECTED`，尚未执行完整 Stage 2/E02。
本文件中的确认串只是精确 CLI 契约，不构成当前对远程连接、上传、正式训练或
E02 formal open 的授权。

服务器访问必须遵守 `docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md`：只从受控环境变量读取连接
事实，不打印凭据，不执行环境变量中的原始命令；先运行固定 `TARCA_SERVER_PROBE_OK` 探针，
再执行用户当前明确授权的白名单操作，并在 `finally` 清理临时密钥、known_hosts、代理脚本
和残留进程。

## 2. 开机后的第一条实验入口

先把下列三个文件通过既有安全通道上传到新服务器的同一目录：

```text
tarca-stage2-v1-server.tar.gz
tarca-stage2-v1-server.tar.gz.sha256
tarca-stage2-v1-server.tar.gz.receipt.json
```

在服务器校验并解压，目标目录应位于至少有 300 GiB 可用空间的本地 NVMe：

```bash
sha256sum --check tarca-stage2-v1-server.tar.gz.sha256
mkdir -p /opt/tarca
tar -xzf tarca-stage2-v1-server.tar.gz -C /opt/tarca
cd /opt/tarca
```

构建固定镜像；正式来源从包内 capsule 导入，不从网络获取。若基础镜像和 Node 构建镜像未在
宿主机缓存，镜像层的首次拉取仍需要可信 registry 通路。

```bash
docker compose -f deploy/stage2/compose.yaml build
```

运行唯一 preflight。`N` 是此刻距整机清空的真实剩余小时数，不是套餐总时长：

```bash
docker compose -f deploy/stage2/compose.yaml run --rm tarca-stage2 \
  bootstrap --mode preflight --remaining-rental-hours N
```

镜像内的等价固定入口是：

```bash
bash deploy/stage2/bootstrap.sh --mode preflight --remaining-rental-hours N
```

该步骤不会训练或打开 formal 数据。它验证环境、两卡、来源和 checkpoint，并让两张 4090
同时运行固定 PatchTST/iTransformer 代表性 workload，估计未缩减完整工作量的 critical path。
只有 `estimated_remaining_seconds + 3600 < N * 3600` 时才返回 `PREFLIGHT_PASS`。超时、OOM、
来源漂移、磁盘/RAM 不足或非有限数值均 fail closed；不要通过删 seed、降 epoch 或减 trajectory
绕过。

## 3. 监控先行

preflight 通过后可以启动只读监控服务：

```bash
docker compose -f deploy/stage2/compose.yaml up -d tarca-stage2
docker compose -f deploy/stage2/compose.yaml ps
```

服务只绑定 `127.0.0.1:8765`。从操作端建立 SSH 隧道后访问：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<server>
```

页面只展示执行状态、进度、资源、checkpoint 新鲜度、ETA 和告警；运行中不泄露模型比较或
formal 科学中间结论。

## 4. Stage 2 独立点火边界

必须在用户看到 preflight 证据并另行明确授权后，才可使用下面的精确确认串。确认串为
`I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN`。建议以脱离 SSH 会话的命名容器启动：

```bash
docker compose -f deploy/stage2/compose.yaml run -d \
  --name tarca-stage2-run tarca-stage2 stage2 launch \
  --acknowledgement I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN
```

状态查询不会打开科学结果：

```bash
docker compose -f deploy/stage2/compose.yaml run --rm tarca-stage2 stage2 status
docker logs --tail 200 tarca-stage2-run
```

首次 launch 创建稳定 run identity 和 SQLite；数据库存在后禁止再次 launch。进程中断时保留
`artifacts/stage2/`，使用同一科学图 resume：

```bash
docker compose -f deploy/stage2/compose.yaml run -d \
  --name tarca-stage2-resume tarca-stage2 stage2 resume \
  --acknowledgement I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN
```

## 5. 24 小时清空与恢复

`TARCA_24H_RESET` 表示第 24 小时整机断电并清空；Docker 命名卷、宿主机 bind mount 和容器
层都不能被当作跨机器持久化。`TARCA_RESET_MARGIN` 固定为 1 小时：点火时剩余租期必须严格
大于实测 ETA 加 1 小时，否则等待一台新的完整租期服务器。

任何异常或接近 reset margin 时，先生成内容寻址恢复 capsule：

```bash
docker compose -f deploy/stage2/compose.yaml run --rm tarca-stage2 stage2 recover
```

随后把 `artifacts/stage2/runtime/recovery/` 中最新目录和全部已发布 artifact 下载到本地或持久
对象存储，并核对 recovery receipt。新机器上恢复同一路径，重新构建镜像、重新运行 preflight，
再使用 `resume`；不得用新 launch 隐藏旧失败。

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
