# TARCA E02 v1 全新服务器交接手册

日期：2026-09-01

> 后续状态（2026-09-01）：本文件是 E02 点火前的历史交接手册。E02 已完成 `COMPLETED / PASS`；
> 正式指标、收据身份和下游边界以 `docs/auth/TARCA_E02_HANDOFF_SNAPSHOT_2026-09-01.md` 为准。

## 1. 这套交接现在能做什么

本地实施已经把 E02 点火前的工作收成两件上传文件：

1. Stage 2 完整结果归档，提供已经冻结的模型、开发数据、执行账本和收据；
2. 最新服务器 bundle，提供代码、固定配置、离线依赖、双卡探针和启动脚本。

在一台全新的目标服务器上，准备脚本会自动完成：校验两个上传文件、核对当前代码、恢复
Stage 2、验证所有冻结模型、检查服务器硬件、同时测试两张 GPU、估算 E02 用时、检查剩余租期、
运行 E02 prepare/dry-run/preflight。最后停在 `E02_READY_FOR_USER_LAUNCH`。

它不会创建 formal grant，不会生成 120 条正式轨迹，不会启动 E02，不会显示科学结果。正式
launch 仍需要用户在看到预检证据后另行明确授权。

## 2. 固定输入

| 文件 | 固定身份 |
|---|---|
| `tarca-stage2-v1-complete-20260901T011423Z.tar.gz` | SHA-256 `7d77ca6bd7d09b3fd9abd7814ef169f11814e774ec479a04d6f1a1c751fe966a` |
| 同名 `.sha256` | 必须与归档一起上传 |
| `tarca-stage2-v1-server.tar.gz` | 以同目录最新 `.sha256` 为权威；包内不能自引用自身哈希 |
| 同名 `.sha256` | 必须与 bundle 一起上传 |
| Stage 2 freeze receipt | 内部 SHA-256 `37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166` |
| E02 scientific config | SHA-256 `9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c` |

不要再上传旧的 device-mismatch recovery 包来启动 E02。那个包只用于 Stage 2 事故恢复；现在
Stage 2 已经 37/37 完成并冻结，E02 应使用上述 complete archive。

## 3. 服务器配置结论

目标镜像 `pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04` 和
`2×RTX 4090 24GB、28C、224GB RAM` 与冻结 profile 一致。实际机器仍必须满足：

- Python 3.10、PyTorch 2.2.2、CUDA 12.1；
- 恰好两张名称含 4090 的 GPU，每张驱动报告显存不少于 23 GiB；
- 至少 28 个物理 CPU 核、224 GiB 总内存；
- artifact 所在本地磁盘至少 200 GiB 可用，推荐 300 GiB NVMe；
- 实测 E02 ETA 加 1 小时必须严格小于服务器剩余租期。

任何一项不满足都会停止，不会减少模型、seed、轨迹、窗口、bootstrap 次数或判定门槛。

## 4. 功能层流程

1. **验货**：确认上传的两个大文件没有损坏，也确认当前服务器目录确实来自这个 bundle。
2. **恢复 Stage 2**：只恢复 `artifacts/stage2/`；拒绝目录穿越、链接、未知目录和覆盖不同内容。
3. **确认冻结模型**：核对 Stage 2 数据、normalizer、四个基线和六个神经 checkpoint 的完整哈希链。
4. **检查机器**：确认镜像版本、两张 4090、CPU、内存和磁盘达到门槛。
5. **实测两卡**：GPU 0 与 GPU 1 同时加载前两个冻结 iTransformer；完成后 GPU 0 测第三个。
6. **估算租期**：把开发集实测速率外推到 120 条轨迹、51,000 个正式窗口，加 35% 安全余量和固定运维余量。
7. **签发预检收据**：把归档、bundle、配置、冻结收据、硬件、探针、ETA 和租期截止时间绑定在一起。
8. **停在授权线前**：输出 `E02_READY_FOR_USER_LAUNCH`，等待用户单独确认。

## 5. Docker 宿主机路径

先按服务器 access runbook 完成连接与上传，再校验并解开服务器 bundle：

```bash
sha256sum --check tarca-stage2-v1-server.tar.gz.sha256
tar -xzf tarca-stage2-v1-server.tar.gz -C /opt/tarca
cd /opt/tarca
```

`N` 是此刻距整机清空的真实剩余小时数：

```bash
bash deploy/stage2/e02_bootstrap.sh \
  --stage2-archive /path/to/tarca-stage2-v1-complete-20260901T011423Z.tar.gz \
  --server-bundle /path/to/tarca-stage2-v1-server.tar.gz \
  --remaining-rental-hours N
```

脚本构建固定镜像，在容器中执行同一套恢复和预检，然后退出。成功标志只有：

```text
E02_READY_FOR_USER_LAUNCH
```

## 6. 已经位于目标容器内的路径

若 SSH 落点本身已经是目标 PyTorch 容器且没有 Docker daemon，不要安装或嵌套 Docker：

```bash
bash deploy/stage2/e02_bootstrap_direct.sh \
  --repository-root /opt/tarca \
  --stage2-archive /path/to/tarca-stage2-v1-complete-20260901T011423Z.tar.gz \
  --server-bundle /path/to/tarca-stage2-v1-server.tar.gz \
  --remaining-rental-hours N
```

该入口使用包内 wheelhouse 和带 `--system-site-packages` 的隔离环境，继承镜像已有的
PyTorch/CUDA，不联网安装，不覆盖系统环境。两条路径产出的 preflight 收据具有同一合同。

## 7. 并行调度如何使用硬件

“吃满”表示依赖允许时把所有有用任务放进安全预算，不运行占位计算。

| 阶段 | 有用并发 |
|---|---|
| formal 数据生成 | 24 个 CPU 工作进程；其余 4 核留给调度/监控与系统 I/O |
| 第一预测波 | GPU 0 一个 iTransformer、GPU 1 一个 iTransformer，同时用 8 CPU 跑 strongest-linear |
| 第二预测波 | 先空闲的 GPU 执行第三个 iTransformer；已经完成的预测立刻触发 CPU score 回填 |
| score | 每个 4 CPU/24 GiB；依赖满足且预算允许时并发，最多四个有用 score 波槽 |
| bootstrap | 8 CPU/48 GiB，等待四组 score 完整后运行 |
| decision/receipt | 轻量 CPU 串行收口 |

GPU 任务是一卡一 worker、每个声明 20 GiB 显存；不在无 NVLink 的双 4090 上强制 DDP。主机
总准入上限固定为 24 个工作核和 200 GiB RAM，另保留 1 核调度/监控、3 核系统/I/O。短暂空闲
若来自依赖等待就是正常状态，不用虚假任务填充。

## 8. 单独授权后的点火与监控

只有用户在预检之后再次明确批准，才使用冻结确认串启动：

```bash
docker compose -f deploy/stage2/compose.yaml run -d \
  --name tarca-e02-run tarca-stage2 e02 launch \
  --acknowledgement I_ACKNOWLEDGE_E02_V1_FORMAL_RUN
```

E02 数据库出现后，可另起只读监控：

```bash
docker compose -f deploy/stage2/compose.yaml run -d --service-ports \
  -e TARCA_RUNTIME_DATABASE=/opt/tarca/artifacts/e02/runtime/execution.sqlite3 \
  -e TARCA_EXECUTION_KIND=e02-v1 --name tarca-e02-monitor tarca-stage2 monitor
```

服务只通过 `127.0.0.1:8765` 和符合 runbook 的 SSH 隧道访问，不能暴露到公网。前端只展示任务、
心跳、资源、失败类别和 ETA；实验完成前不展示 CRPS、coverage 或门禁结果。

## 9. 当前边界

本地已完成代码、测试、离线包和两条服务器入口。由于本机没有双 4090/CUDA，真实硬件、吞吐和
租期探针仍标记为 `NOT_RUN_LOCAL_NO_CUDA`；它们只能在用户提供的服务器上执行。E02 formal
grant、120 条正式轨迹、预测、评分、bootstrap、判定和 receipt 均仍为 `NOT_RUN_E02_FORMAL`。
