# Stage1B v2 官方运行时构建与服务器交接报告

> 构建状态：`BUILT_NOT_QUALIFIED`
> 活动科学系列：`v2`
> 报告日期：2026-08-26
> 当前分支：`codex/stage1b-v2-official-runtime`
> 回退分支：`codex/stage1b-v2-pre-official-runtime`（`2bc422b`）

## 1. 功能层结论

Stage1B v2 的 17 个实施任务已经构建完成。当前仓库可以在指定的双 RTX 4090 服务器上：

1. 下载并锁定六个官方仓库到精确 commit；
2. 先复现官方生成器和模型关键行为，再生成资格数据与生成器自带真值；
3. 把完整资格编译成 74 个不可变任务，并在两张 GPU 上并行调度；
4. 失败后从已验证制品和 checkpoint 恢复，不重跑已经完成的科学任务；
5. 通过只读中文前端显示进程、期望/实际 CPU、内存、双卡利用率/显存和 ETA；
6. 把最终结果绑定到来源、环境、精度、硬件、运行图、任务清单和执行计划哈希；
7. 首次冻结为 `v2-r1`，用户授权覆盖时新增下一不可变 v2 修订并移动活动指针。

这不表示资格已经通过。构建期间没有运行完整 Stage1B，没有运行 E01/E02，也没有创建
冻结修订。

## 2. 实际构建的流程

```text
来源初始化（可写独立容器）
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

来源初始化与科学运行被故意拆开。`stage1b-source-init` 以可写命名卷下载来源；正式
`stage1b` 服务使用同一命名卷的只读挂载，因此训练进程不能覆盖官方代码。

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
| Python 3.10 服务器兼容矩阵（除 Stage0 doctor CLI） | 361/361 通过，155.53 秒 |
| Python 3.10 coverage 复核 | 361/361 通过；branch coverage 80.13%，达到 80% 门槛 |
| Python 3.11 Stage0 doctor/CLI | 5/5 通过；冷启动曾一次超过 60 秒，立即复跑 25.80 秒通过 |
| Ruff lint | 全仓通过 |
| Ruff format | 全仓 145 个文件通过 |
| mypy strict | 73 个源文件通过 |

覆盖率复核之后新增的两个冷启动用例只增加覆盖：独立来源初始化服务，以及 Git fetch 的
两次有界瞬态重试。最终 361 用例服务器兼容矩阵已包含并通过这两个用例。

### 监控前端

- Vitest：14/14；
- statements 97.39%，branches 84.93%，functions 97.56%，lines 100%；
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
- 全新空命名卷的六来源下载、commit、文件哈希与树哈希验证：通过；
- 固定哈希 Python 锁文件直接审计：没有已知漏洞；
- 秘密模式扫描：未发现私钥、API key、密码或 token；
- Stage1A 检查：`PASS`，未触碰正式数据、未训练；
- Stage1B 检查：`UNFROZEN`，与真实状态一致；
- `docs/auth`：无修改。

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
- 能访问六个固定官方 Git 仓库。

调度器为监控保留 1 核、为系统和 I/O 保留 3 核，其余最多 24 核分配给数据与任务；两个
独立神经任务默认分别占用一张 GPU，并依据显存、利用率、数据等待和 OOM 证据调整装箱，
不通过改 seed、样本、epoch 或门槛来换速度。

## 6. 服务器执行命令

在服务器仓库根目录执行：

```bash
git switch codex/stage1b-v2-official-runtime
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml build
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm stage1b-source-init
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm stage1b preflight
```

`preflight` 必须确认双卡、28 核、224GB、PyTorch/CUDA/Python、官方来源、FP32/AMP 精度和
24 小时外推全部通过。若外推超过 24 小时，默认停止；只有用户再次明确授权时才使用
`preflight --authorize-over-24-hours`。

探针通过后启动完整资格：

```bash
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm --service-ports stage1b launch
```

浏览器打开 `http://127.0.0.1:8765`。另一个终端可以只读查询：

```bash
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm stage1b status
```

主进程被中断后，不得重新 `launch`，应恢复：

```bash
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm --service-ports stage1b resume
```

## 7. 资格后冻结

只有 74 个任务全部完成、套件 Gate 为 `PASS`、完整失败台账已审阅后，才运行首次冻结：

```bash
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm \
  --entrypoint python stage1b scripts/run_stage1b_qualification.py freeze \
  --series v2 --revision-id v2-r1
```

以后如用户授权覆盖，必须重新执行受影响资格并新增下一修订，例如：

```bash
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm \
  --entrypoint python stage1b scripts/run_stage1b_qualification.py freeze \
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
