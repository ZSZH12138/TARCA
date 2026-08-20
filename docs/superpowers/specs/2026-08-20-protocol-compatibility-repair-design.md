# TARCA Protocol Compatibility Repair Design

## Goal

Resolve the two protocol inconsistencies found at the Stage 0 to Stage 1A boundary without changing the scientific identity or rewriting the already frozen Stage 0 artifacts.

## Scope

This repair has two narrowly bounded parts:

1. Make the protocol's `Sha256Hash` notation match the canonical wire format already used by the code and every frozen Stage 0 artifact: exactly 64 lowercase hexadecimal characters, without a `sha256:` prefix.
2. Define the previously referenced but unspecified `SealedAccessGrant`, and make the Stage 1A physical-read function signatures accept an optional grant that is checked before sealed data is read.

The repair does not implement dataset registries, loaders, window construction, SCM generation, model training, or any later Stage 1 capability.

## Compatibility Decision

The protocol document revision becomes `v2.0.1`, while the stable protocol identity remains `TARCA-E2E-STAGE-PROTOCOL-2.0`. This is a compatibility erratum: it aligns the written notation with the existing wire representation and therefore does not change any stored hash, `ArtifactRef`, research-contract reference, Gate 0 decision, or completion receipt.

A change-control record in `docs/auth/` will state the evidence, affected clauses, compatibility reasoning, and rollback boundary. No frozen Stage 0 artifact will be overwritten or regenerated.

## Sealed Access Contract

The public contract surface will add these strict, frozen types:

```python
class DatasetWindowPartition(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    TEST_SEEN_REGIME = "TEST_SEEN_REGIME"
    TEST_UNSEEN_REGIME = "TEST_UNSEEN_REGIME"

class DatasetSpec(StrictContractModel):
    name: str
    version: str

class AccessScope(StrictContractModel):
    sealed: bool
    scope_name: str

class SealedAccessGrant(StrictContractModel):
    grant_id: str
    dataset: DatasetSpec
    scope_name: str
    allowed_partitions: tuple[DatasetWindowPartition, ...]
    authorization_ref: ArtifactRef
    issued_at: UtcDatetime
    expires_at: UtcDatetime
```

Validation rules:

- identifiers and scope names must be non-blank;
- dataset names and versions are logical keys, not paths;
- allowed partitions must be non-empty and unique;
- `expires_at` must be later than `issued_at`;
- the authorization reference must have artifact type `SEALED_ACCESS_AUTHORIZATION`;
- the grant must match the exact dataset, scope and requested partition before any sealed physical read;
- an absent, expired or mismatched grant fails closed;
- a grant authorizes reading only; it never authorizes fitting on validation/test data or changes scientific identity.

The protocol signatures for `build_windows()` and `hash_dataset()` will add:

```python
grant: SealedAccessGrant | None = None
```

Actual loaders remain a Stage 1A implementation task. This repair supplies the contract and validation helper only.

## Files

- Modify `docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md`.
- Create `docs/auth/TARCA_PROTOCOL_CHANGE_CONTROL_CCP_0001.md`.
- Create `src/tarca/contracts/data_access.py`.
- Modify `src/tarca/contracts/__init__.py`.
- Add a small focused regression test file under `tests/stage0/`.

## Testing

Use a small TDD set that proves:

- raw 64-character hashes remain valid and prefixed hashes are rejected;
- sealed grants are strict and immutable;
- invalid dataset keys, duplicate partitions, wrong authorization type and invalid time bounds are rejected;
- the access validator rejects missing, expired or mismatched grants and accepts an exact valid grant.

After the focused tests pass, run the existing full test suite, Stage 0 aggregate gate, Ruff, formatting, Mypy and lock-file check. No network or server access is required.

## Non-goals

- no Stage 1A registry or loader implementation;
- no Arrow/Parquet schema work;
- no dependency changes;
- no Stage 0 artifact replacement;
- no Gate 0 or novelty reassessment;
- no server connection, data download, model training, SCM, intervention, OT or DRO.
