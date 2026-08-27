# Stage1B v2 官方运行时构建与服务器交接报告

> 构建状态：`BUILT_NOT_QUALIFIED`
> 活动科学系列：`v2`
> 报告日期：2026-08-27
> 当前分支：`codex/stage1b-runtime-supervision-fix`
> 原始运行时回退点：`codex/stage1b-v2-official-runtime`（`f2104f7`）

## 1. 功能层结论

Stage1B v2 的 17 个实施任务已经构建完成。当前仓库可以在指定的双 RTX 4090 服务器上：

1. 在本地下载、审核并打包六个官方仓库到精确 commit；服务器离线导入并复核，不自行访问
   GitHub；
2. 先复现官方生成器和模型关键行为，再生成资格数据与生成器自带真值；
3. 把完整资格编译成 74 个不可变任务，把完整计划写入状态库，并在两张 GPU 上安全并行；
4. 失败后从已验证制品和 checkpoint 恢复，不重跑已经完成的科学任务；
5. 每 2 秒采集真实主机、进程和 NVML 数据，通过只读中文前端显示进程、期望/实际 CPU、
   内存、双卡利用率/显存、采样新鲜度和有证据的 ETA；
6. 把最终结果绑定到来源、环境、精度、硬件、运行图、任务清单和执行计划哈希；
7. 首次冻结为 `v2-r1`，用户授权覆盖时新增下一不可变 v2 修订并移动活动指针。

这不表示资格已经通过。构建期间没有运行完整 Stage1B，没有运行 E01/E02，也没有创建
冻结修订。

## 2. 实际构建的流程

```text
本地来源审核与 Git bundle 打包
→ 安全上传源码包与 receipt
→ 服务器离线导入到可写来源卷
→ 正式容器只读校验来源
→ 双卡/CPU/内存/版本/精度/24 小时硬件探针
→ 官方复现
→ 74 个资格任务
→ 世界健康与四分区数据
→ VAR + PatchTST + iTransformer
→ 冻结模型可操作性与盲比较
→ 自动门槛与完整失败台账
→ 哈希绑定资格收据
→ 用户审阅
→ v2-r1 冻结
```

来源导入与科学运行被故意拆开。本地使用精确 commit、关键资产与完整树哈希审核来源，导出
Git bundle、manifest 和外层 SHA-256 receipt；服务器的 `stage1b-source-import` 只从本地
bundle 重建深度一的 detached checkout，再重复同一组校验。正式 `stage1b` 服务使用同一
命名卷的只读挂载，因此训练进程不能覆盖官方代码，也不会在运行期间访问 GitHub。

## 3. 科学与实施边界

- 活动世界是线性 VAR 控制、单尺度 Lorenz-96 F=10/F=40、双尺度 Lorenz-96、GVAR
  捕食者—猎物和修正 CML；
- 主资格只要求至少一个独立主机制家族满足预注册门槛，不要求每个 seed/horizon 都赢；
- 每个主世界要求至少 40 个完整轨迹比较单元、CRPS 胜率至少 65%、总体 skill 为正、
  seen/unseen 均多数获胜，并通过 NLL、MAE、校准、worst-regime 与概率尺度护栏；
- 65% 是 TARCA v2 的预注册准入线，不冒充学界统一阈值；
- 真值由世界生成器在生成事实/反事实轨迹时一并产生，预测模型不能生成或修改真值；
- 资格只有 `QUAL_TRAIN`、`QUAL_TUNE`、`QUAL_SEEN`、`QUAL_UNSEEN`；保留正式 seed、
  `TEST`、E01 和 E02 会被边界校验拒绝；
- 调度器只读取任务状态与资源，不读取 CRPS、胜率、truth 或 Gate 结果。

详细规范见 `docs/research/stage1b_world_qualification_spec.md`。

## 4. 本地验证结果

### Python 双环境矩阵

项目主合同要求 Python 3.11/3.12，服务器镜像固定 Python 3.10，因此采用两个互补矩阵：

| 矩阵 | 结果 |
|---|---|
| Python 3.10 服务器兼容矩阵（除 Stage0 doctor CLI） | 380/380 通过，221.31 秒 |
| Python 3.10 coverage 复核 | 380/380 通过；branch coverage 80.39%，达到 80% 门槛 |
| Python 3.11 Stage0 doctor/CLI | 5/5 通过；并行重负载时冷启动曾超过 60 秒，单独复跑 20.16 秒通过 |
| Ruff lint | 全仓通过 |
| Ruff format | 全仓 145 个文件通过 |
| mypy strict | 73 个源文件通过 |

380 用例包含活动资源账本、完整运行计划、真实监督器、监控投影和正式运行器接入的新回归
测试；没有执行完整 Stage1B、E01 或 E02。

### 监控前端

- Vitest：17/17；
- statements 97.61%，branches 86.36%，functions 97.72%，lines 100%；
- Vite 生产构建：通过；
- Playwright/Edge：1/1；
- `npm audit --omit=dev`：0 个已知漏洞。

ECharts 已拆成独立动态分块。它仍产生一个大于 500KB 的性能提示，但不阻断功能、安全或
主应用首包。

### 容器与供应链

- Compose 配置校验：通过；
- 镜像构建：通过；
- 最终用户：`tarca`（UID 10001），根文件系统只读；
- 精确运行时：Python 3.10.14、PyTorch 2.2.2、CUDA 12.1；
- 空状态烟测：`{"status":"EMPTY"}`；
- 本地审核的六来源源码包从空缓存离线导入、commit、文件哈希与树哈希验证：通过；
- 固定哈希 Python 锁文件直接审计：没有已知漏洞；
- 秘密模式扫描：未发现私钥、API key、密码或 token；
- Stage1A 检查：`PASS`，未触碰正式数据、未训练；
- Stage1B 检查：`UNFROZEN`，与真实状态一致；
- `docs/auth`：无修改。

本地 WSL 没有可见 NVIDIA adapter，因此带双 GPU reservation 的 Compose 容器会在 NVIDIA
prestart hook 按预期拒绝启动；同一镜像在非 root、只读根文件系统、无 GPU 的直接空状态烟测
返回 `{"status":"EMPTY"}`。双卡 Compose 启动和真实 NVML 数值仍属于服务器验收项。

Windows 上让 `pip-audit` 创建临时安装环境会因 Linux 专用 `uvloop` 失败；改用完全固定锁
文件的 `--no-deps --disable-pip --strict` 模式后审计通过，Docker 的 Linux 构建也证明该
锁文件可按哈希完整安装。

## 5. 服务器硬件合同

服务器必须至少满足当前授权合同：

- 2 张 NVIDIA RTX 4090，每张 24GB 显存；
- 至少 28 个物理 CPU 核；
- 至少 224GB 内存；
- NVIDIA Container Toolkit 可用；
- 建议至少 100GB 可用本地高速存储；
- 可通过既有安全通道接收本地审核的源码包与 receipt；正式服务器不需要访问 GitHub。

调度器为监控保留 1 核、为系统和 I/O 保留 3 核，其余最多 24 核分配给数据与任务；主机
任务内存合同上限为 200 GiB，并要求至少 100 GiB 可用本地存储。每轮调度都会从总容量中
扣除所有 `RUNNING` 尝试已经绑定的 CPU、内存和 GPU；资源不足时任务继续排队，释放后下一
轮补位。声明 20 GiB 显存的神经训练/冻结任务默认每张 24 GiB GPU 只运行一个，不做无证据
的激进超额装箱，也不通过改 seed、样本、epoch 或门槛来换速度。

## 5.1 真实监控与 ETA 语义

- 运行创建时登记完整 74 节点不可变计划，因此前端总数包含尚未入队的未来节点；
- 正式调度器使用同一个 `PsutilNvmlTelemetryProbe`，监督器按 2 秒节流写入 run 级主机/GPU
  样本及带 `attempt_id` 的进程样本；
- 监控失败不会生成伪 0 样本；无样本显示“遥测不可用”，样本年龄不超过 10 秒显示“数据
  正常”，更旧样本保留最后数值并显示“数据过期”；
- 训练任务 ETA 只使用真实进程开始时间、最新进度时间和 `completed_steps / total_steps`；
  同 phase、同 model 的已完成真实耗时可用于未启动任务，没有足够证据时保持“校准中”；
- API、WebSocket 和前端全部只读，不提供停止、重启、修改参数或改写资格结果的入口。

## 6. 服务器执行命令

先在本地仓库根目录生成并审核源码包：

```powershell
.\.venv\Scripts\python.exe scripts/package_stage1b_source_capsule.py
```

本次已本地验证的源码包为：

- `stage1b-v2-official-sources.tar.gz`
- capsule SHA-256：`cb508aaaa7fa8aaccc380b09bdbf167c4e04b9b38ac0d01abbcf4862c53880ea`
- manifest SHA-256：`9b233b370832f1cf1dffc9bb6c2fc034cc3e1f0732ddf0431219d2cb7849a79c`

将该文件和同名 `.receipt.json` 通过 runbook 规定的安全通道上传到服务器。然后在服务器
仓库根目录执行：

```bash
export PYTHONPATH="$PWD/deploy/stage1b/py310:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TARCA_STAGE1B_SOURCE_MODE=offline-capsule
export TARCA_STAGE1B_SOURCE_CACHE_ROOT="${TARCA_STAGE1B_SOURCE_CACHE_ROOT:-$PWD/third_party/stage1b}"
export TARCA_STAGE1B_DATABASE="${TARCA_STAGE1B_DATABASE:-$PWD/artifacts/stage1b/runtime/execution.sqlite3}"
export TARCA_STAGE1B_STATIC_ROOT="${TARCA_STAGE1B_STATIC_ROOT:-$PWD/frontend/stage1b-monitor/dist}"

python scripts/import_stage1b_source_capsule.py \
  --capsule /secure-transfer/stage1b-v2-official-sources.tar.gz \
  --receipt /secure-transfer/stage1b-v2-official-sources.tar.gz.receipt.json \
  --cache-root "$TARCA_STAGE1B_SOURCE_CACHE_ROOT"

bash deploy/stage1b/entrypoint.sh preflight
```

预检通过后启动完整资格：

```bash
bash deploy/stage1b/entrypoint.sh launch
```

`preflight` 必须确认双卡、28 核、224GB、PyTorch/CUDA/Python、官方来源、FP32/AMP 精度和
24 小时外推全部通过。若外推超过 24 小时，默认停止；只有用户再次明确授权时才使用
`preflight --authorize-over-24-hours`。

服务器本机浏览器可以打开 `http://127.0.0.1:8765`。远程电脑先建立 SSH 隧道：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<server>
```

然后在远程电脑浏览器打开同一地址。另一个服务器终端可以只读查询：

```bash
python scripts/run_stage1b_runtime.py status
```

主进程被中断或服务器重启后，SQLite 状态、已验证制品和 checkpoint 仍保存在
运行目录中。不得重新 `launch`，应恢复：

```bash
bash deploy/stage1b/entrypoint.sh resume
```

## 7. 资格后冻结

只有 74 个任务全部完成、套件 Gate 为 `PASS`、完整失败台账已审阅后，才运行首次冻结：

```bash
python scripts/run_stage1b_qualification.py freeze \
  --series v2 --revision-id v2-r1
```

以后如用户授权覆盖，必须重新执行受影响资格并新增下一修订，例如：

```bash
python scripts/run_stage1b_qualification.py freeze \
  --series v2 --revision-id v2-r2 --authorize-override \
  --prior-revision-id v2-r1 --authorization-reason "用户批准的覆盖原因"
```

不得覆盖 `r1`，不得把失败运行冻结为历史版本，也不得把实施修订伪装成 v3 科学系列。

## 8. 仍需在服务器完成的验收

1. 实际双 4090 `preflight` 收据；
2. 24 小时内的真实 ETA 与资源利用率；
3. 完整 74 任务 Stage1B 资格；
4. 全部世界、模型、seed、regime、horizon 比较行和失败原因审阅；
5. 通过后创建 `v2-r1`；
6. Stage1B 冻结以后，另行授权 E01/E02。

在这些步骤发生前，项目状态必须保持 `BUILT_NOT_QUALIFIED` / `UNFROZEN`。
