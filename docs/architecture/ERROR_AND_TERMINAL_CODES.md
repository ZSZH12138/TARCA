# Error and Terminal Codes

## Error taxonomy

| Code | Meaning |
| --- | --- |
| `PROTOCOL_ERROR` | protocol/config identity or semantic binding is invalid |
| `CONTRACT_ERROR` | typed input/output contract is invalid |
| `DATA_ERROR` | data content, split, or leakage constraint is invalid |
| `SCIENTIFIC_FAIL` | a frozen scientific gate failed; do not reinterpret it as infrastructure failure |
| `RESOURCE_BLOCKED` | runtime resource admission failed before safe execution |
| `RUNTIME_ERROR` | execution failed independently of scientific meaning |
| `ARTIFACT_INVALID` | artifact hash/schema/completion marker is invalid |
| `AUTHORIZATION_BLOCKED` | explicit grant is missing or invalid |
| `SEALED_ACCESS_VIOLATION` | sealed/truth payload crossed a forbidden boundary |
| `ARCHITECTURE_VIOLATION` | module dependency or contract ownership rule was broken |
| `UNIMPLEMENTED_CAPABILITY` | future capability is intentionally not implemented |

## Frozen terminal vocabulary

The E02d-r4 continuation retains these terminal states:

`P9P10_PROTOCOL_INVALID`, `P9P10_RESOURCE_BLOCKED`, `SUSPENDED_RESUMABLE`,
`PHASE9_SCIENTIFIC_GATE_FAIL`, `PHASE10_NOT_RUN_BY_GATE`,
`PHASE10_CONFIRMATORY_FAIL`, `E02D_R4_CANONICAL_INVALID`, and
`E02D_R4_CANONICAL_PASS`.

Architecture-only skeleton calls terminate with the error code
`UNIMPLEMENTED_CAPABILITY`; they never emit a scientific `PASS`.

## State separation

`TaskState` is operational: `PENDING -> LEASED -> RUNNING -> PUBLISHING ->
COMPLETED`, with failure/retry bookkeeping in `TaskAttemptRecord`.
Operational failure does not change scientific identity. A scientific gate
failure is not repaired by rerunning, averaging, or changing the architecture.
