# TARCA Stage 2 收尾、交接与仓库发布设计

日期：2026-09-01

状态：用户已确认方案 A；桌面汇报与 Word 文件不在本任务范围内

## 1. 目标

本任务把已经完成并冻结的 Stage 2 实施事实同步到说明文件和 `docs/auth/` 权威交接层，清除可再生中间物，审计并整理当前 Stage 2 相关代码，形成可供后续任务直接接手的详细快照，最后以当前分支安全更新远端 `main`。

本任务不改变 Stage 2 科学配置、模型选择、冻结身份、E02 formal 边界或任何预注册阈值；不重新运行服务器实验；不创建桌面汇报或 Word 文件。

## 2. 已知事实与固定身份

- 当前工作分支：`codex/e01-server-ready`。
- 设计开始时本地 HEAD：`a79284b6958a5d30b48506905710680299d9fb9c`。
- 设计开始时远端 `main`：`d4d6a41ed989fe8ccb5682939aa2f6ad6ff826ce`。
- 远端 `main` 是当前分支的祖先，当前历史允许快进；推送仍使用精确 lease 防止覆盖设计后出现的未知远端提交。
- Stage 2 run：`run-acff24d96653a25d4aac54b9389c605d8c35293cc930f9fa8a560947306401fb`。
- 固定图最新状态：`37/37 COMPLETED`。
- 历史记录：六个 `attempt-1 / WORKER_ERROR` 保留；对应六个 `attempt-2 / COMPLETED` 为当前完成状态。
- Stage 2 状态：`FROZEN`；`formal_access_event_count = 0`。
- Stage 2 scientific SHA-256：`c2df021d248c2ffcdcf6133179f4b88c86ea88ae4e3f72630f302b88402e0e32`。
- freeze receipt 内部身份：`37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166`。
- strongest linear：`VAR`。
- primary iTransformer seed：`1797287582`。
- E02：未准备、未预检、未授权、未运行；不存在 E02 formal grant 或结果。
- 最终完整服务器归档 SHA-256：`7d77ca6bd7d09b3fd9abd7814ef169f11814e774ec479a04d6f1a1c751fe966a`。
- 当前 Stage 2 服务器 bundle SHA-256：`1d05dd8a98178ef111990131b682552e5b9cd51e1b23f79397bc4a4fec99deee`。

## 3. 文档同步设计

### 3.1 状态说明层

修改 `README.md` 的 Stage 2 / E02 当前状态：

- Stage 2 从旧的 `LOCAL_IMPLEMENTATION_COMPLETE / NOT_RUN` 更新为服务器运行已完成和 `FROZEN`；
- 明确 37/37、同 run 恢复、E02 未运行；
- 指向新的权威 Stage 2 快照和现有服务器运行报告；
- 仓库范围说明同步为 Stage1B `FROZEN_V2`、E01 `v2/PASS`、Stage 2 `FROZEN`、E02 未运行。

### 3.2 规范与计划层

保持规范文件的职责边界：

- `docs/auth/TARCA_项目计划书.md`：只增加当前实施事实的权威快照入口，不修改研究问题、候选贡献、证据等级或 Gate；
- `docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md`：继续保持“实施状态中立”，只把当前状态指针从 E01 快照扩展到 Stage 2 快照；
- `docs/auth/TARCA_具体实施计划.md`：在 Stage 2 / E02 部分增加 2026-09-01 实施同步，记录 Stage 2 已冻结以及 E02 仍需独立授权；修改末尾已过期的“后续从 Stage 2 开始”为“后续从 E02 开始”；
- `docs/auth/TARCA_E01_HANDOFF_SNAPSHOT_2026-08-30.md`：只增加顶部后续状态提示，保留 E01 历史正文；
- `docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md`：保留本次已经写入的最小直接容器/独立运行说明，不扩大为 Stage 2 专用手册。

对具有 ReadOnly 属性的权威文件，编辑前只临时移除 ReadOnly，写入后恢复原属性。

### 3.3 研究实施记录层

- `docs/research/stage2_e02_local_implementation_report_v1.md`：增加完成状态更新，明确正文描述的是服务器恢复前的本地实施基线；
- `docs/research/stage2_e02_server_handoff_v1.md`：保留现有完成提示和事故恢复步骤，不把审计历史改写成当前待办；
- `docs/research/stage2_server_run_report_v1.md`：作为服务器完成事实来源保留；
- `docs/research/stage2_device_mismatch_recovery_v1.md`：作为事故恢复规范保留。

## 4. Stage 2 权威快照设计

新建 `docs/auth/TARCA_STAGE2_HANDOFF_SNAPSHOT_2026-09-01.md`，并在完成核验后设置 ReadOnly。快照至少包含：

1. 功能层结论与当前项目位置；
2. Stage1B、E01、Stage 2、E02 的严格关系；
3. 固定 run、科学配置、来源和冻结身份；
4. 37 个任务的完成口径、六次旧失败和六次恢复尝试的解释；
5. strongest-linear 与 primary-iTransformer 的冻结选择；
6. 设备不一致事故中“预期、实际、原因、修复”的完整记录；
7. 双 RTX 4090 并行调度和只读前端监督的实现边界；
8. 最终归档、恢复归档、服务器 bundle、来源 capsule 的本地保管位置和哈希；
9. GitHub 可发布内容与本地专有实验产物的边界；
10. 验证命令和交接检查表；
11. E02 的正确起点、独立确认串边界和不得继承 Stage 2 授权的规则；
12. 已知限制：Stage 2 冻结不等于 E02 PASS，也不授权 Stage 3/4。

快照不记录 SSH 地址、用户名、私钥、代理值或任何临时服务器凭据。

## 5. 机器可核验冻结凭证

从已经校验的最终服务器归档中复制两个小型 JSON 到规范位置：

```text
artifacts/stage2/frozen/v1/stage2_freeze_receipt.json
artifacts/stage2/frozen/v1/stage2_manifest.json
```

发布前验证：

- freeze receipt 外层文件 SHA-256 为 `5ec77ab844ef0bc793bf8543db57f01856ab603718a21fd1a19c42bf0947d8e5`；
- manifest 外层文件 SHA-256 为 `6d9ef496956a714e956c57800f0c1cf479a042624f757f54f4882a99f8d132d4`；
- receipt 内部 `manifest_sha256`、`receipt_sha256`、`scientific_sha256` 可由现有 Stage 2 校验逻辑重算；
- 两个文件不包含模型权重、原始数据、服务器路径或凭据。

## 6. 本地保留与删除边界

### 6.1 必须保留但不得上传 GitHub

- 最终完整服务器归档及 `.sha256` sidecar；
- 事故前固定恢复归档及 `.sha256` sidecar；
- 当前可用的 Stage 2 服务器 bundle、`.sha256` 和 receipt；
- Stage 2 官方来源 capsule 及 receipt；
- Stage1B、E01 的既有正式本地产物和第三方来源缓存。

以上内容用于重算、事故审计或服务器重放，继续由本地文件系统保管。

### 6.2 可删除的中间物

在验证其替代物存在并校验通过后，删除以下精确目标：

- 最终完整归档旁的 `extracted/` 解压副本；
- `tarca-stage2-v1-server.verify-a.tar.gz`；
- `tarca-stage2-v1-server.verify-b.tar.gz`；
- `artifacts/test-tmp-stage2-source-import/`；
- `artifacts/stage2/verification-local/`；
- `.mypy_cache/`、`.pytest_cache/`、`.ruff_cache/` 和仓库内所有 `__pycache__/`；
- 前端验证完成后的 `frontend/stage1b-monitor/node_modules/`、`test-results/`、`playwright-report/` 和可再生 `dist/`。

预计释放约 1.18 GB。删除前必须验证每个目标的绝对路径都位于 `C:\Users\DELL\Desktop\TARCA` 内；不得使用未解析的递归通配删除。

## 7. `.gitignore` 与上传边界

将 Stage 2 规则调整为默认忽略 `artifacts/stage2/**`，仅允许：

```text
artifacts/stage2/frozen/v1/stage2_freeze_receipt.json
artifacts/stage2/frozen/v1/stage2_manifest.json
```

服务器归档、结果、bundle、来源 capsule、运行数据库、checkpoint、日志、解压副本和验证临时目录均不得暂存。

发布时使用显式路径清单执行 `git add`；禁止 `git add .`。暂存后必须检查：

- `git diff --cached --name-only`；
- `git diff --cached --stat`；
- 暂存文件大小；
- 常见密钥、令牌、私钥头、服务器连接串和临时路径；
- 不存在被忽略的大型实验产物。

## 8. 代码清洗设计

代码审计范围是当前分支相对于远端 `main` 的 Stage 2、E02、execution、monitoring、前端、部署脚本和对应测试。

删除代码必须满足至少一项证据：

- Ruff 报告未使用导入、未使用局部变量、重复定义或未解析名称；
- TypeScript 编译器或 ESLint 报告不可达/未使用代码；
- 全仓引用搜索证明私有函数或常量无消费者；
- 两段恢复逻辑语义和错误处理完全相同，且提取公共函数不改变接口；
- 过期调试输出、临时分支或注释与现行契约冲突。

当前初筛 `F401,F811,F821,F841` 已通过。因此不得为了满足“清洗”形式而删除有用恢复代码。若进一步审计没有发现可证明的死代码，只进行格式、命名、过期状态注释和小型重复条件整理，并在最终汇报中如实说明没有发现可安全删除的功能代码。

任何功能性清理都必须先有或补充覆盖该行为的测试，然后运行受影响测试与完整测试套件。

## 9. 验证设计

发布前执行并保存退出状态：

1. Stage 2 冻结 receipt/manifest 重算与哈希检查；
2. 完整 Python pytest；
3. 完整 Ruff；
4. mypy；
5. Stage 0 与 Stage 1A 检查入口；
6. 前端 Vitest、TypeScript/build 与 Playwright E2E；
7. Stage 2 文档契约和 bundle/container 静态契约测试；
8. Bash 脚本 `bash -n`；
9. `git diff --check`；
10. Git 暂存内容、文件大小、忽略规则和敏感信息审计。

若完整测试或发布审计失败，不提交最终收尾 commit，不推送远端 `main`，保留工作树供修复。

## 10. Git 发布设计

1. 先单独提交本设计规格；
2. 按后续实施计划完成文档、凭证、忽略规则、代码清洗和测试；
3. 使用显式路径暂存最终源代码、测试、配置和文档；
4. 创建 conventional commit；
5. 推送前重新读取远端 `main`；
6. 使用绑定已读取远端 SHA 的 lease 更新：

```text
git push --force-with-lease=refs/heads/main:<fresh-remote-main-sha> origin HEAD:refs/heads/main
```

如果远端 `main` 在最终读取后发生变化，lease 必须拒绝推送；不得改用无 lease 的强制推送覆盖未知提交。

## 11. 完成标准

- README、研究实施记录和 `docs/auth/` 对 Stage 2 状态描述一致；
- 新 Stage 2 权威快照完整、无凭据、ReadOnly；
- 两个小型冻结 JSON 可独立核验并进入 Git；
- 大型实验产物全部保持未跟踪且被忽略；
- 可再生中间物已按精确路径删除，正式归档和复现包仍存在；
- 代码审计有证据，未进行无关重构；
- 所有要求的验证通过；
- 远端 `main` 精确指向当前收尾分支最终提交；
- 未创建任何桌面汇报或 Word 文件。
