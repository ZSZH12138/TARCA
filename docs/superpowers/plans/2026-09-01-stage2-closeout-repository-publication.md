# TARCA Stage 2 Closeout and Repository Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. TARCA project rules require inline single-agent execution and prohibit subagent dispatch.

**Goal:** Synchronize the completed Stage 2 state across project documentation, publish only its small frozen evidence, remove reproducible intermediates, perform evidence-based code cleanup, and update remote `main` from the current branch without uploading large or sensitive artifacts.

**Architecture:** Preserve normative research documents as status-neutral specifications and put concrete run facts in a new authoritative handoff snapshot. Keep full experiment and recovery archives local, expose only two verified JSON receipts to Git, enforce this boundary through `.gitignore` and tests, and use an exact remote lease for publication.

**Tech Stack:** Python 3.11, pytest, Ruff, mypy, Pydantic, React 19, TypeScript 7, Vitest, Playwright, Git, PowerShell, Bash.

**Spec:** `docs/superpowers/specs/2026-09-01-stage2-closeout-repository-publication-design.md`

## Global Constraints

- Execute inline in the current TARCA task; do not use subagents.
- Do not change Stage 2 scientific configuration, frozen selections, hashes, formal-access count, or E02 authorization boundaries.
- Do not create a desktop report, DOCX, PDF, or any Word artifact.
- Do not upload checkpoints, databases, logs, archives, bundles, source capsules, third-party sources, credentials, or extracted server results.
- Preserve the final complete archive, the pre-recovery archive, the current server bundle, and the official source capsule locally.
- Use `D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe` with `PYTHONPATH=C:\Users\DELL\Desktop\TARCA\src` for Python verification.
- Use `apply_patch` for repository file edits.
- Temporarily clear and then restore ReadOnly only for protected authoritative files being edited.
- Use the explicit `git add --` path list in Task 7; never use `git add .`.
- Publish with a lease bound to the SHA returned by the immediately preceding `git ls-remote`, and stop if the lease fails.

---

### Task 1: Add Stage 2 closeout documentation and publication contract tests

**Files:**
- Modify: `tests/stage2/test_documentation_contract.py`
- Test: `tests/stage2/test_documentation_contract.py`

**Interfaces:**
- Consumes: repository Markdown, `.gitignore`, and canonical Stage 2 frozen JSON files.
- Produces: tests that fail until the authority snapshot, current status, evidence files, and upload boundary exist.

- [ ] **Step 1: Add closeout constants and the frozen-evidence test**

Add imports and constants:

```python
import json
from hashlib import sha256
from pathlib import Path

from tarca.stage2.freeze import Stage2FreezeReceipt
from tarca.stage2.manifest import stage2_manifest_from_payload

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/research/stage2_e02_local_implementation_report_v1.md"
HANDOFF = ROOT / "docs/research/stage2_e02_server_handoff_v1.md"
SNAPSHOT = ROOT / "docs/auth/TARCA_STAGE2_HANDOFF_SNAPSHOT_2026-09-01.md"
FREEZE_ROOT = ROOT / "artifacts/stage2/frozen/v1"
```

Add:

```python
def test_stage2_frozen_evidence_is_small_valid_and_publishable() -> None:
    receipt_path = FREEZE_ROOT / "stage2_freeze_receipt.json"
    manifest_path = FREEZE_ROOT / "stage2_manifest.json"
    receipt_bytes = receipt_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()

    assert len(receipt_bytes) < 10_000
    assert len(manifest_bytes) < 20_000
    assert sha256(receipt_bytes).hexdigest() == (
        "5ec77ab844ef0bc793bf8543db57f01856ab603718a21fd1a19c42bf0947d8e5"
    )
    assert sha256(manifest_bytes).hexdigest() == (
        "6d9ef496956a714e956c57800f0c1cf479a042624f757f54f4882a99f8d132d4"
    )
    receipt = Stage2FreezeReceipt.model_validate_json(receipt_bytes)
    manifest = stage2_manifest_from_payload(json.loads(manifest_bytes))
    assert receipt.status == "FROZEN"
    assert receipt.formal_access_event_count == 0
    assert receipt.scientific_sha256 == manifest.scientific_sha256
```

- [ ] **Step 2: Add documentation synchronization assertions**

Add:

```python
def test_stage2_closeout_status_is_synchronized() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    snapshot = SNAPSHOT.read_text(encoding="utf-8")
    implementation = (ROOT / "docs/auth/TARCA_具体实施计划.md").read_text(
        encoding="utf-8"
    )
    protocol = (
        ROOT / "docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md"
    ).read_text(encoding="utf-8")

    for text in (readme, snapshot, implementation):
        assert "37/37 COMPLETED" in text
        assert "E02" in text
    assert "TARCA_STAGE2_HANDOFF_SNAPSHOT_2026-09-01.md" in readme
    assert "TARCA_STAGE2_HANDOFF_SNAPSHOT_2026-09-01.md" in protocol
    assert "NOT_RUN_E02_FORMAL" in snapshot
    assert "37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166" in snapshot
```

- [ ] **Step 3: Add static upload-boundary assertions**

Add:

```python
def test_stage2_gitignore_exposes_only_small_frozen_evidence() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/artifacts/stage2/**" in ignore
    assert "!/artifacts/stage2/frozen/v1/stage2_freeze_receipt.json" in ignore
    assert "!/artifacts/stage2/frozen/v1/stage2_manifest.json" in ignore
    assert "/artifacts/stage2/server-results/" not in ignore
```

The final assertion ensures the new blanket rule replaces, rather than coexists with, a partial server-results exception.

- [ ] **Step 4: Run the focused test and verify RED**

Run:

```powershell
$env:PYTHONPATH='C:\Users\DELL\Desktop\TARCA\src'
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m pytest tests/stage2/test_documentation_contract.py -q
```

Expected: FAIL because the new snapshot and canonical Stage 2 frozen evidence do not yet exist and README still states Stage 2 was not run.

---

### Task 2: Publish the two small frozen artifacts and enforce the Git boundary

**Files:**
- Modify: `.gitignore`
- Create: `artifacts/stage2/frozen/v1/stage2_freeze_receipt.json`
- Create: `artifacts/stage2/frozen/v1/stage2_manifest.json`
- Test: `tests/stage2/test_documentation_contract.py`

**Interfaces:**
- Consumes: verified files inside the final complete server archive extraction.
- Produces: canonical small Git-publishable Stage 2 frozen evidence while every large Stage 2 artifact remains ignored.

- [ ] **Step 1: Add the Stage 2 default-local ignore policy**

Replace the existing partial Stage 2 artifact rules with:

```gitignore
# Stage 2 defaults to local custody; Git publishes only the two small frozen v1 receipts.
/artifacts/stage2/**
!/artifacts/stage2/
!/artifacts/stage2/frozen/
/artifacts/stage2/frozen/**
!/artifacts/stage2/frozen/v1/
/artifacts/stage2/frozen/v1/**
!/artifacts/stage2/frozen/v1/stage2_freeze_receipt.json
!/artifacts/stage2/frozen/v1/stage2_manifest.json
```

Keep the existing generic `/artifacts/test-tmp-*/` rule.

- [ ] **Step 2: Copy the exact verified receipt bytes with `apply_patch`**

Create `stage2_freeze_receipt.json` with the exact canonical JSON line from the final archive:

```json
{"formal_access_event_count":0,"manifest_sha256":"ff50bb15819dea13bd0f31cdb3fc331f02b2ed528509022b0d1aa676d3d8e5d2","primary_itransformer_seed":1797287582,"receipt_sha256":"37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166","schema_version":"tarca-stage2-freeze-v1","scientific_sha256":"c2df021d248c2ffcdcf6133179f4b88c86ea88ae4e3f72630f302b88402e0e32","status":"FROZEN","strongest_linear_model_id":"VAR"}
```

Create `stage2_manifest.json` with the exact canonical JSON bytes already present in:

```text
artifacts/stage2/server-results/stage2-v1-complete-20260901T011423Z/extracted/artifacts/stage2/frozen/v1/stage2_manifest.json
```

Do not reserialize, pretty-print, or reorder the manifest.

- [ ] **Step 3: Verify exact hashes and ignore behavior**

Run:

```powershell
Get-FileHash artifacts\stage2\frozen\v1\stage2_freeze_receipt.json -Algorithm SHA256
Get-FileHash artifacts\stage2\frozen\v1\stage2_manifest.json -Algorithm SHA256
git check-ignore -v artifacts/stage2/server-results/stage2-v1-complete-20260901T011423Z/tarca-stage2-v1-complete-20260901T011423Z.tar.gz
git check-ignore artifacts/stage2/frozen/v1/stage2_freeze_receipt.json
```

Expected: hashes `5ec77ab...` and `6d9ef496...`; the large archive is ignored; `git check-ignore` returns non-zero for the allowed receipt.

---

### Task 3: Synchronize explanatory, research, and authoritative documents

**Files:**
- Modify: `README.md`
- Modify: `docs/auth/TARCA_项目计划书.md`
- Modify: `docs/auth/TARCA_具体实施计划.md`
- Modify: `docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md`
- Modify: `docs/auth/TARCA_E01_HANDOFF_SNAPSHOT_2026-08-30.md`
- Preserve existing modification: `docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md`
- Modify: `docs/research/stage2_e02_local_implementation_report_v1.md`
- Preserve existing modification: `docs/research/stage2_e02_server_handoff_v1.md`
- Preserve: `docs/research/stage2_server_run_report_v1.md`
- Create: `docs/auth/TARCA_STAGE2_HANDOFF_SNAPSHOT_2026-09-01.md`
- Test: `tests/stage2/test_documentation_contract.py`

**Interfaces:**
- Consumes: Stage 2 final run report, frozen receipt/manifest, recovery specification, and the normative authority hierarchy.
- Produces: one consistent current-status narrative and one detailed authoritative handoff snapshot without altering scientific rules.

- [ ] **Step 1: Temporarily clear ReadOnly on the protected files**

Run exact attribute changes only for files that currently have ReadOnly:

```powershell
$protected = @(
  'docs/auth/TARCA_项目计划书.md',
  'docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md'
)
foreach ($path in $protected) {
  $item = Get-Item -LiteralPath $path
  $item.IsReadOnly = $false
}
```

- [ ] **Step 2: Update README current status**

Replace the stale Stage 2 paragraph with a concise statement containing:

```text
Stage 2 v1 已在双 RTX 4090 服务器上完成并冻结为 `FROZEN`。固定 run 的最新任务状态为
`37/37 COMPLETED`；六个 attempt-1 `WORKER_ERROR` 作为事故历史保留，六个同 run attempt-2
均已完成。E02 为 `NOT_RUN_E02_FORMAL`，没有打开 formal 数据，也没有 E02 PASS/FAIL 结论。
```

Link both the new authority snapshot and `docs/research/stage2_server_run_report_v1.md`.

- [ ] **Step 3: Add minimal authority pointers**

Add one current-implementation pointer near the metadata/authority section of `TARCA_项目计划书.md`:

```text
> 当前实施事实不改写本计划的研究规则；Stage 2 完成身份与 E02 交接边界见
> `TARCA_STAGE2_HANDOFF_SNAPSHOT_2026-09-01.md`。
```

Extend the implementation-status-neutral paragraph in the protocol so that current facts point to both the E01 and Stage 2 authority snapshots. Do not alter the Stage 2 I/O or Exit clauses.

Add a top status note to the E01 snapshot:

```text
> 2026-09-01 后续状态：Stage 2 已完成并冻结；E02 尚未运行。当前交接入口见
> `TARCA_STAGE2_HANDOFF_SNAPSHOT_2026-09-01.md`。本文其余内容继续作为 E01 完成时快照保留。
```

- [ ] **Step 4: Add Stage 2 actual-execution synchronization to the implementation plan**

After the Stage 2/E02 sections, record:

```text
实施同步（2026-09-01）：Stage 2 v1 已沿用固定 run 完成同运行恢复，最新图状态为
`37/37 COMPLETED`，并发布状态为 `FROZEN` 的 receipt；strongest linear 固定为 `VAR`，
primary iTransformer seed 固定为 `1797287582`，formal access event count 为 0。
E02 仍为 `NOT_RUN_E02_FORMAL`，不得把 Stage 2 授权延伸为 E02 formal 授权。
```

Replace the late-document stale next-step sentence so the next scientific action is E02 preparation/preflight and independent authorization, not Stage 2.

- [ ] **Step 5: Make the local implementation report historically explicit**

Add a top block stating that Stage 2 later reached `37/37 COMPLETED / FROZEN`, while the report body intentionally preserves the pre-recovery local implementation state. Keep all historical `NOT_RUN_RECOVERY_ON_NEW_SERVER` assertions as scoped historical evidence rather than deleting them.

- [ ] **Step 6: Write the authoritative Stage 2 snapshot**

Create the snapshot with these exact main sections:

```markdown
# TARCA Stage 2 完成情况与任务交接快照

## 1. 功能层结论
## 2. Stage1B、E01、Stage 2 与 E02 的关系
## 3. 固定运行与冻结身份
## 4. 实验任务、模型和选择结果
## 5. 设备不一致事故与受控恢复
## 6. 双 GPU 调度与只读前端监督
## 7. 最终归档与本地保管边界
## 8. GitHub 发布边界
## 9. 当前代码、配置和操作入口
## 10. 独立验证证据
## 11. 已知边界
## 12. 下一任务的正确起点
## 13. 后续任务交接检查表
```

Populate every section from the frozen identities in the spec and the run/recovery reports. State explicitly that Stage 2 completion validates the forecasting suite freeze only; it does not constitute E02 PASS or authorize Stage 3/4.

- [ ] **Step 7: Restore ReadOnly and set the new snapshot ReadOnly**

Run:

```powershell
$protected = @(
  'docs/auth/TARCA_项目计划书.md',
  'docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md',
  'docs/auth/TARCA_STAGE2_HANDOFF_SNAPSHOT_2026-09-01.md'
)
foreach ($path in $protected) {
  $item = Get-Item -LiteralPath $path
  $item.IsReadOnly = $true
}
```

- [ ] **Step 8: Run the focused closeout test and verify GREEN**

Run the Task 1 focused pytest command.

Expected: all documentation contract tests pass.

---

### Task 4: Remove the verified redundant runtime assignment and complete code audit

**Files:**
- Modify: `src/tarca/stage2/runtime.py`
- Test: `tests/stage2/test_runtime.py`

**Interfaces:**
- Consumes: existing `_compiled_graph()` tests and Ruff findings.
- Produces: the same `Stage2Graph` return value without the redundant local assignment; an evidence-backed decision not to refactor high-risk recovery paths merely for style.

- [ ] **Step 1: Verify existing runtime tests cover graph compilation**

Run:

```powershell
rg -n "compiled_graph|launch_stage2|resume_stage2|compile_stage2_graph" tests/stage2/test_runtime.py tests/stage2/test_cli.py
```

Expected: launch/resume/CLI paths exercise `_compiled_graph()` through public runtime commands.

- [ ] **Step 2: Remove the Ruff-confirmed redundant assignment**

Change:

```python
graph = compile_stage2_graph(config, Stage2GraphInputs(...))
return graph
```

to:

```python
return compile_stage2_graph(config, Stage2GraphInputs(...))
```

Do not refactor the C901 recovery functions: their branch structure enforces fail-closed validation, and the audit found no unused imports, unused locals, duplicate definitions, or unresolved names.

- [ ] **Step 3: Run the affected runtime and CLI tests**

Run:

```powershell
$env:PYTHONPATH='C:\Users\DELL\Desktop\TARCA\src'
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m pytest tests/stage2/test_runtime.py tests/stage2/test_cli.py -q
```

Expected: PASS with no behavioral change.

- [ ] **Step 4: Run targeted cleanliness checks**

Run:

```powershell
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m ruff check src scripts tests --no-cache --select F401,F811,F821,F841,RET504
```

Expected: no findings.

---

### Task 5: Run complete verification on the final tracked tree

**Files:**
- Verify: all tracked source, tests, deployment scripts, frontend, and documentation.

**Interfaces:**
- Consumes: final implementation tree before generated dependencies are removed.
- Produces: fresh evidence supporting completion and publication.

- [ ] **Step 1: Verify Stage 2 evidence hashes and scientific receipt**

Run the two `Get-FileHash` commands from Task 2 and the focused documentation contract test.

- [ ] **Step 2: Run full Python tests**

Run:

```powershell
$env:PYTHONPATH='C:\Users\DELL\Desktop\TARCA\src'
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m pytest -q
```

Expected: zero failures.

- [ ] **Step 3: Run Python quality checks**

Run:

```powershell
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m ruff check . --no-cache
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m mypy
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' scripts/check_stage0.py
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' scripts/check_stage1a.py --json
```

Expected: all exit zero; Stage 0 and Stage 1A report PASS.

- [ ] **Step 4: Run frontend tests and production build**

From `frontend/stage1b-monitor` run:

```powershell
npm test -- --run
npm run coverage
npm run build
npm run e2e
```

Expected: Vitest, coverage threshold, TypeScript/Vite build, and Playwright all pass.

- [ ] **Step 5: Verify shell and repository hygiene**

Run:

```powershell
wsl bash -n deploy/stage2/bootstrap.sh
wsl bash -n deploy/stage2/entrypoint.sh
wsl bash -n deploy/stage2/recovery_bootstrap.sh
wsl bash -n deploy/stage2/recovery_bootstrap_direct.sh
git diff --check
```

Expected: all exit zero. A WSL proxy warning may be reported but is not a syntax failure.

---

### Task 6: Delete exact reproducible intermediates after verification

**Files:**
- Delete local-only generated directories and files listed in the spec; no tracked file is deleted.

**Interfaces:**
- Consumes: verified final archive, sidecar, current bundle, and lockfiles.
- Produces: a leaner local workspace while retaining audit and reproduction evidence.

- [ ] **Step 1: Verify retained replacements before deletion**

Resolve and verify these exact files:

```text
artifacts/stage2/server-results/stage2-v1-complete-20260901T011423Z/tarca-stage2-v1-complete-20260901T011423Z.tar.gz
artifacts/stage2/server-results/stage2-v1-complete-20260901T011423Z/tarca-stage2-v1-complete-20260901T011423Z.tar.gz.sha256
artifacts/stage2/server-archives/tarca-stage2-recovery-20260831T102151Z.tar.gz
artifacts/stage2/server-archives/tarca-stage2-recovery-20260831T102151Z.tar.gz.sha256
artifacts/stage2/server-bundles/tarca-stage2-v1-server.tar.gz
artifacts/stage2/server-bundles/tarca-stage2-v1-server.tar.gz.sha256
artifacts/stage2/server-bundles/tarca-stage2-v1-server.tar.gz.receipt.json
artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz
artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz.receipt.json
```

Recheck the final archive and current bundle hashes against the fixed values in the spec.

- [ ] **Step 2: Resolve and validate every deletion target**

Use PowerShell `Resolve-Path -LiteralPath` for each existing target and assert its full path starts with `C:\Users\DELL\Desktop\TARCA\`. Build the deletion list only from explicit literal paths and enumerated `__pycache__` directory objects.

- [ ] **Step 3: Delete the validated targets with native PowerShell**

Use `Remove-Item -LiteralPath` on each validated file or directory. Do not pass the list to another shell, do not use globs, and do not recursively delete any computed path that failed the workspace-prefix assertion.

- [ ] **Step 4: Measure cleanup and recheck retained evidence**

Confirm all deletion targets are absent, all retained files remain present, and record before/after byte totals for the final report.

---

### Task 7: Audit staging, commit the closeout, and update remote main

**Files:**
- Stage: every intended source, test, config, deployment, documentation, and the two small Stage 2 frozen JSON files.
- Exclude: all ignored/local-only experiment artifacts and generated dependencies.

**Interfaces:**
- Consumes: verified final tracked tree and fresh remote-main identity.
- Produces: one conventional closeout commit and a remote `main` pointing to the final current-branch commit.

- [ ] **Step 1: Stage only explicit intended paths**

Use explicit `git add --` arguments for:

```text
.gitignore
README.md
artifacts/stage2/frozen/v1/stage2_freeze_receipt.json
artifacts/stage2/frozen/v1/stage2_manifest.json
configs/stage2/stage2_device_mismatch_recovery_v1.json
deploy/stage2/
docs/auth/
docs/research/stage2_device_mismatch_recovery_v1.md
docs/research/stage2_e02_local_implementation_report_v1.md
docs/research/stage2_e02_server_handoff_v1.md
docs/research/stage2_server_run_report_v1.md
docs/superpowers/plans/2026-09-01-stage2-closeout-repository-publication.md
frontend/stage1b-monitor/
scripts/prepare_e01_v2_server_bundle.py
scripts/prepare_stage2_v1_server_bundle.py
scripts/run_stage2_v1.py
src/tarca/e02/
src/tarca/execution/
src/tarca/monitoring/
src/tarca/stage1b/
src/tarca/stage2/
tests/e01/
tests/e02/
tests/execution/
tests/monitoring/
tests/stage2/
```

The already committed design spec is not restaged unless it changed.

- [ ] **Step 2: Audit staged names, sizes, ignored artifacts, and secrets**

Run:

```powershell
git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
git status --short --ignored
```

For each staged file, assert its worktree size is below 1 MiB. Search the staged diff for private-key headers, common token prefixes, password assignments, SSH connection commands containing concrete hosts, and `C:\Users\DELL` absolute paths. Review every match as either forbidden or a documented non-secret test fixture.

- [ ] **Step 3: Commit the verified closeout**

Run:

```powershell
git commit -m "feat: finalize stage2 frozen handoff"
```

- [ ] **Step 4: Re-read the remote main SHA immediately before push**

Run:

```powershell
git ls-remote origin refs/heads/main
```

Assign the returned 40-character SHA to `$freshRemoteMainSha`. Verify it remains an ancestor of the final local HEAD or equals the previously inspected remote SHA. If it is an unknown new commit, stop and report instead of overwriting it.

- [ ] **Step 5: Push the current branch to remote main with an exact lease**

Run:

```powershell
$freshRemoteMainSha = (git ls-remote origin refs/heads/main).Split()[0]
git push "--force-with-lease=refs/heads/main:$freshRemoteMainSha" origin HEAD:refs/heads/main
```

Expected: remote `main` updates to the final local HEAD. If the lease rejects, stop; never retry with plain `--force`.

- [ ] **Step 6: Verify remote and local final identities**

Run:

```powershell
git rev-parse HEAD
git ls-remote origin refs/heads/main
git status --short
```

Expected: local HEAD equals remote `main`; `git status --short` shows no unintended tracked or untracked upload candidates. Local ignored formal archives remain present.
