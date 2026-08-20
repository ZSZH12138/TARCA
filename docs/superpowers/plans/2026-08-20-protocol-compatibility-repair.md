# TARCA Protocol Compatibility Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. This repository requires inline single-agent execution; subagent execution is prohibited. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the hash notation with frozen Stage 0 artifacts and add a complete fail-closed sealed-access grant contract.

**Architecture:** Keep the stable protocol identity at 2.0 and publish a patch-level document erratum plus CCP record. Add small strict/frozen data-access contracts and one pure validation function; do not implement loaders or any other Stage 1A capability.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, Markdown protocol documents.

**Spec:** `docs/superpowers/specs/2026-08-20-protocol-compatibility-repair-design.md`

## Global Constraints

- Preserve all frozen Stage 0 JSON bytes and hashes.
- Keep `TARCA-E2E-STAGE-PROTOCOL-2.0` as the stable protocol identity.
- Canonical SHA-256 wire values remain exactly 64 lowercase hexadecimal characters.
- Sealed reads fail before physical I/O unless a matching, current grant exists.
- Do not implement registries, loaders, windows, Arrow schemas, SCMs, models, interventions, OT or DRO.
- Execute inline in the current agent only.

---

### Task 1: Data-access contracts and validator

**Files:**
- Create: `src/tarca/contracts/data_access.py`
- Modify: `src/tarca/contracts/__init__.py`
- Test: `tests/stage0/test_protocol_compatibility.py`

**Interfaces:**
- Consumes: `StrictContractModel`, `UtcDatetime`, `ArtifactRef`.
- Produces: `DatasetWindowPartition`, `DatasetSpec`, `AccessScope`, `SealedAccessGrant`, `validate_sealed_access(...) -> None`.

- [ ] **Step 1: Write focused failing tests**

Add tests that use `TypeAdapter(Sha256Hash)` to prove raw hashes pass and prefixed hashes fail; construct a valid grant; assert strict/frozen behavior; and assert missing, expired or mismatched grants raise `PermissionError`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\stage0\test_protocol_compatibility.py -q
```

Expected: collection/import failure because the new access contracts do not exist.

- [ ] **Step 3: Implement the minimal contracts**

Implement the five partition enum values, logical-key validation for dataset name/version, non-blank scope/grant identifiers, non-empty unique partitions, exact authorization artifact type, increasing UTC validity interval, and this validator signature:

```python
def validate_sealed_access(
    dataset: DatasetSpec,
    partition: DatasetWindowPartition,
    access: AccessScope,
    grant: SealedAccessGrant | None,
    accessed_at: datetime,
) -> None: ...
```

For `access.sealed=False`, return without requiring a grant. For sealed access, reject an absent grant, dataset/scope/partition mismatch, non-UTC access time, access before issuance, or access at/after expiry.

- [ ] **Step 4: Export the contracts and verify GREEN**

Run the focused test command again. Expected: all focused tests pass.

---

### Task 2: Protocol erratum and change-control record

**Files:**
- Modify: `docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md`
- Create: `docs/auth/TARCA_PROTOCOL_CHANGE_CONTROL_CCP_0001.md`
- Modify test: `tests/stage0/test_protocol_compatibility.py`

**Interfaces:**
- Consumes: approved design and the access contract from Task 1.
- Produces: protocol document revision 2.0.1 with stable identity 2.0 and documented Stage 1A optional grant parameters.

- [ ] **Step 1: Add a failing protocol consistency test**

Assert that the protocol contains the exact raw-hash notation, `SealedAccessGrant` declaration, optional grant parameters for both physical-read functions, and a reference to `CCP-0001`.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: failure because the protocol still states the prefixed hash and omits the grant declaration.

- [ ] **Step 3: Apply the approved documentation repair**

Set the document revision to 2.0.1 while retaining stable protocol identity 2.0; correct the hash row; define the grant fields and invariants; add optional grant parameters to `build_windows()` and `hash_dataset()`; strengthen Stage 1A acceptance; and create the CCP record explaining why no Stage 0 artifact is regenerated.

- [ ] **Step 4: Run the focused test and verify GREEN**

Expected: all focused tests pass.

---

### Task 3: Full verification

**Files:**
- No additional production files.

**Interfaces:**
- Consumes: completed Tasks 1–2.
- Produces: fresh verification evidence.

- [ ] **Step 1: Run Stage 0 and quality gates**

```powershell
.\.venv\Scripts\python.exe scripts\check_stage0.py --json
.\.venv\Scripts\python.exe -m pytest --cov=tarca --cov-report=term --cov-fail-under=80 -q
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\ruff.exe format --check src tests scripts
.\.venv\Scripts\mypy.exe src
D:\software\MyAnaconda\envs\tarca-local-py311\python.exe -m uv lock --check
```

Expected: every command exits 0, all tests pass and coverage remains at least 80%.

- [ ] **Step 2: Review the exact diff and repository state**

Confirm that no frozen file under `artifacts/stage0/` changed and that the pre-existing untracked handoff snapshot remains untouched.
