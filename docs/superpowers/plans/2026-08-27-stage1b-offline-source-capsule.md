# Stage1B Offline Source Capsule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` in inline mode. TARCA project rules prohibit subagents, so every step must be executed by the current primary agent.

**Goal:** Package locally audited official sources into a verifiable artifact that a reopened server imports and uses without GitHub access.

**Architecture:** Create a focused source-capsule module that builds Git bundles and an integrity manifest from existing verified checkouts, then imports them atomically into the configured source cache. Thread one explicit offline acquisition mode through preflight, graph jobs, and official model adapters; preserve all scheduler and monitoring interfaces.

**Tech Stack:** Python 3.10/3.11, Git bundle, SHA-256, canonical JSON, tar archive, Pytest, existing Stage1B runtime CLI.

**Spec:** `docs/superpowers/specs/2026-08-27-stage1b-offline-source-capsule-design.md`

## Global Constraints

- Work only on `codex/stage1b-runtime-supervision-fix`; never modify `docs/auth/**`.
- Execute inline; do not create or use subagents.
- Do not run full Stage1B, E01, E02, or freeze a revision.
- Preserve current dual-GPU resource admission, CPU affinity handling, scheduler, checkpoints, monitoring API and frontend behavior.
- Online acquisition remains permitted only for local capsule construction; `offline-capsule` must make no remote Git request.
- Each production behavior change follows RED → GREEN → REFACTOR and full Python coverage remains at least 80%.

---

### Task 1: Specify and Test Offline Acquisition

**Files:**
- Modify: `src/tarca/stage1b/sources.py`
- Test: `tests/stage1b/test_sources.py`

**Interfaces:**
- Produces `SourceAcquisitionMode.ONLINE` and `.OFFLINE_CAPSULE`.
- Changes `materialize_source(source, cache_root, runner, *, mode=...)`.

- [ ] Write a test whose cache is absent in offline mode and whose runner records calls.
- [ ] Run the test; expect failure because the existing function fetches.
- [ ] Implement a mode parser and fail closed before temporary checkout creation when offline cache is absent.
- [ ] Verify an existing clean checkout is still validated locally in offline mode and no fetch/remote command occurs.
- [ ] Run `tests/stage1b/test_sources.py`.

### Task 2: Build and Import an Auditable Capsule

**Files:**
- Create: `src/tarca/stage1b/source_capsules.py`
- Create: `scripts/package_stage1b_source_capsule.py`
- Create: `scripts/import_stage1b_source_capsule.py`
- Test: `tests/stage1b/test_source_capsules.py`

**Interfaces:**
- Produces `build_source_capsule(suite, cache_root, output_path, runner) -> SourceCapsuleReceipt`.
- Produces `import_source_capsule(suite, capsule_path, receipt_path, cache_root, runner) -> tuple[SourceMaterializationReceipt, ...]`.

- [ ] Create a local temporary Git source, materialize it, and write a failing round-trip test for bundle → archive → clean import.
- [ ] Add failing tamper tests for outer archive hash and bundle hash; assert existing cache remains unchanged.
- [ ] Implement canonical manifest, deterministic safe archive layout, receipt SHA checks, local bundle fetch, detached checkout, and atomic cache publish.
- [ ] Run the capsule tests plus the source tests.

### Task 3: Apply One Offline Mode to Every Runtime Consumer

**Files:**
- Modify: `scripts/run_stage1b_runtime.py`
- Modify: `src/tarca/stage1b/jobs.py`
- Modify: `src/tarca/stage1b/modeling/patchtst.py`
- Modify: `src/tarca/stage1b/modeling/itransformer.py`
- Test: `tests/stage1b/test_runtime_cli.py`
- Test: `tests/stage1b/test_jobs_contract.py`
- Test: `tests/stage1b/test_official_patchtst.py`
- Test: `tests/stage1b/test_official_itransformer.py`

**Interfaces:**
- Produces `source_cache_root(repo_root)` and `source_acquisition_mode_from_environment()` shared by runtime consumers.
- Preflight source receipt includes the offline source mode and capsule identity when applicable.

- [ ] Write failing tests that set offline mode/cache override and assert no source consumer uses the repository-default cache or remote materialization.
- [ ] Implement centralized environment parsing and use it in preflight, source tasks, and model context verification.
- [ ] Run the focused runtime, jobs and official-model suites.

### Task 4: Assemble the Actual Local Upload Artifact

**Files:**
- Generated, ignored: `artifacts/stage1b/source-capsules/*.tar.gz`
- Generated, ignored: `artifacts/stage1b/source-capsules/*.receipt.json`

- [ ] Materialize all six pinned official sources locally with existing validation.
- [ ] Build the capsule and receipt.
- [ ] Re-import it into an empty temporary cache in offline mode and validate every source.
- [ ] Record only hashes, source IDs and commits in the handoff report; never commit the third-party source bytes.

### Task 5: Protect the Runtime Contract and Hand Off

**Files:**
- Modify: `README.md`
- Modify: `docs/research/stage1b_v2_official_runtime_build_report.md`
- Test: existing execution/monitoring/frontend regression suites

- [ ] Update server instructions to upload, import, preflight and launch with `offline-capsule`.
- [ ] Run full Python tests with coverage, formatter/linter and focused runtime checks.
- [ ] Run frontend install, tests and production build; verify no frontend source was altered unintentionally.
- [ ] Inspect the diff, confirm `docs/auth` remains untouched, commit the implementation, and report the exact local capsule identity and commands required after the server opens.
