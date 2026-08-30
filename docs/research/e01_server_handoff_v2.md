# TARCA E01-v2 服务器即开即用交接

> 当前状态：正式运行已完成并回收，整体 `PASS`。以下流程保留用于审计和必要时按同一身份重放，
> 不是再次运行授权。

目标环境已经冻结为
`pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04`，硬件为单张 RTX 4090 24GB、
14 个物理 CPU 核和 112GB RAM。最低可用磁盘 200GB；建议 300GB 以上。

本文件只规定操作顺序。服务器连接、秘密处理、固定探针和临时文件清理必须遵守
`docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md`。连接、上传/预检、正式点火和最终回收是不同授权，
不能因为能登录就自动扩大操作范围。

## 流程 1：上传前确认包没有损坏

本地交付物是：

```text
artifacts/e01/bundle/tarca-e01-server-v2.tar.gz
artifacts/e01/bundle/tarca-e01-server-v2.tar.gz.sha256
artifacts/e01/bundle/tarca-e01-server-v2.receipt.json
```

压缩包包含 v2 代码、配置、依赖锁、监控前端、Stage1B v2 身份、唯一 v1 历史记录中的 E01-B
证据和逐文件
`SHA256SUMS.json`。它不含 SSH 凭据值、服务器连接事实、正式结果、v1 的 2.5GB 原始 effect
store 或 Windows 用户目录路径。

功能上，这一步是在给“整箱货物”贴封条。上传前和上传后 SHA-256 必须相同；解压后还要逐文件
核对 `SHA256SUMS.json`。任一项不同就停止，不能继续预检。

## 流程 2：连接后只做固定探针

按 runbook 从本机用户环境变量中读取连接事实，白名单解析后建立 SSH。第一次远程命令只能是
固定 `TARCA_SERVER_PROBE_OK` 探针。成功只说明安全通路可用，不代表已经获准上传、安装、预检
或点火。

## 流程 3：上传、解压和五分钟预检

获得本次上传与预检授权后，把三件交付物传到服务器，先核对整包 SHA-256，再解压并进入
`TARCA` 目录。运行：

```bash
bash deploy/e01/server_bootstrap_v2.sh --remaining-rental-hours <实际剩余小时>
```

这条命令依次完成五件事：确认 Python/PyTorch 依赖；核对 E01-B 和 Stage1B 身份；生成固定的
101 个任务但不执行；做不超过五分钟的非正式硬件试跑；选择安全的 CPU 并发和 GPU batch，
估算正式耗时并写入 preflight receipt。

它没有正式点火口令，因此不会误跑 50 个正式种子。预检 PASS 后仍必须停下并向用户报告：
硬件身份、建议 CPU 并发、GPU batch、预计耗时、剩余租期和前端静态件检查结果。

若选择 Docker Compose，可在同一目录构建镜像；正式运行使用
`deploy/e01/compose.e01-v2.yaml`，宿主制品目录必须通过 `TARCA_E01_V2_ARTIFACT_DIR` 绑定，不能
只写进容器层或 Docker 命名卷，也不能挂载 Docker socket。

## 流程 4：用户单独授权后正式点火

只有用户明确给出下面的精确确认，才能继续：

```text
I_ACKNOWLEDGE_E01_V2_FORMAL_RUN
```

直接运行方式：

```bash
bash deploy/e01/server_supervisor_v2.sh launch \
  --acknowledgement I_ACKNOWLEDGE_E01_V2_FORMAL_RUN
```

脚本会把实验和只读前端分别放进独立的 `nohup` 进程，等待 SQLite 建立，再实际访问
`http://127.0.0.1:8765/api/v1/run`。只有进程、数据库和 API 都可用才报告稳定；前端失败会立即
停止刚启动的正式进程。稳定后可以断开 SSH 和关闭 Codex，实验不依赖聊天会话继续运行。

Compose 方式使用同一个精确确认：

```bash
export TARCA_E01_V2_ARTIFACT_DIR=<宿主机持久目录>
docker compose -f deploy/e01/compose.e01-v2.yaml build
docker compose -f deploy/e01/compose.e01-v2.yaml run -d --service-ports e01-v2 \
  launch --acknowledgement I_ACKNOWLEDGE_E01_V2_FORMAL_RUN
```

## 流程 5：用户接管前端监督

前端只绑定服务器本机 `127.0.0.1:8765`，不能直接暴露公网。用户通过 SSH 隧道在本机浏览器
打开同一端口。页面负责回答：任务是否在推进、进程是否活着、CPU/RAM/GPU/显存的预期和实际
占用、ETA 是否可计算、最近检查点是什么、是否出现 OOM/停滞/资源告警。

页面没有停止按钮、远程 shell 或改参数能力，也不会提前显示科学成绩。Codex 确认页面稳定并
把入口交给用户后停止主动轮询；只有用户再次要求诊断、停止、续跑或回收时才介入。

## 流程 6：中断时只续跑，不重跑

若容器、SSH 或服务器意外中断，先保全整个宿主制品目录。恢复同一包、配置、Stage1B 身份、
E01-B 凭据、SQLite 和 artifact store，并重新通过环境校验后，必须再次获得用户授权，然后运行：

```bash
bash deploy/e01/server_supervisor_v2.sh resume \
  --acknowledgement I_ACKNOWLEDGE_E01_V2_FORMAL_RUN
```

`resume` 会跳过已经通过哈希验证的任务。若身份漂移，它会拒绝续跑，不会拿旧结果拼接新实验。

## 流程 7：完成后回收到本地再关服务器

状态查询不会显示中间科学 Gate：

```bash
python scripts/run_e01_v2.py --repository-root . \
  --config configs/e01/e01_v2.yaml \
  --artifact-root artifacts/e01-v2-server status
```

确认 101/101 完成后，按 runbook 精确停止属于本次实验的监控进程，回收整个
`artifacts/e01-v2-server` 目录及服务器生成的 SHA-256 清单。回收内容至少包括 SQLite（以及仍
存在的 WAL/SHM）、完整 artifact store、prepare/preflight 收据、日志、告警和最终聚合制品。

本地逐文件校验、打开最终报告并确认可恢复性以后，才能向用户说“可以关闭服务器”。只看到
页面完成或只下载一份最终 JSON 都不够。

## 流程 8：本次实际执行闭环

本次服务器环境与上面的冻结配置一致。正式点火口令由用户单独给出；运行最终完成 101/101，
失败 0、告警 0。完整目录被归档、生成逐文件 SHA-256 清单并下载到本地；本地重新验证了压缩包、
312 个清单文件、SQLite 任务计数、最终报告内容哈希和总 Gate。服务器监控进程与本地隧道随后
被精确停止，用户已收到可关机结论。

本地保留的原始 v2 压缩包 SHA-256 为
`8b49ea6fafe440876977b7c4a55de11dea39e70ba631ae86430081d286ec855d`，最终报告内容哈希为
`b8f48acfa7ac1fa2bf9790119b1c3a4458035e16f89c058592ef544ea4eaef29`。正式状态以
`artifacts/e01/frozen/v2/qualification_receipt.json` 和权威 E01 快照为准。
