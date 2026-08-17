# TARCA Architecture Contract Freeze v1

## Status and scope

`TARCA_ARCHITECTURE_VERSION = "1.0"`.

This document freezes the logical architecture and public contract boundary. It
does not freeze a scientific algorithm, solver, training schedule, server
topology, or Stage 3 implementation. The architecture is additive and keeps
all pre-existing files at their original paths.

The project remains an internal synthetic/controlled causal-alignment research
system: it studies whether interventions on a dynamic SCM concept can be
represented and detected through a time-series predictor. It does not make
real-market causal claims.

## Authority and grandfathering

The running code, active configs, manifests, receipts, hashes, and tests are
authoritative. Existing Stage 0/1 contracts and active Stage 2/E02d-r4
canonical materials are grandfathered. Historical protocol documents are
evidence only and are not executable authority.

The following existing interfaces are explicitly preserved:

`WindowBatch`, `ForecastDistribution`, `ConceptBatch`, `InterventionSite`,
`InterventionSpec`, `InterventionPair`, `DataManifest`, `RunManifest`,
`MetricRecord`, `ArtifactLayout`, `ForecastPredictor`, and
`ForecastModelAdapter`.

The compatibility baseline is
`artifacts/architecture/CURRENT_PUBLIC_API_BASELINE.json`.

## Three planes

| Plane | Responsibility | Must not do |
| --- | --- | --- |
| Science | data, models, concepts, interventions, effects, localization, robustness, metrics, experiments, existing training | schedule workers, read server state, change governance identity |
| Governance | protocol identity, manifests, grants, sealed access, gates, receipts, terminal states, artifact identity | implement a scientific solver or train a model |
| Execution | scheduler, workers, runtime, resources, retry/recovery, monitoring | alter scientific task identity or model-selection semantics |

Execution attempts are operational records. They may increase after a worker
failure, but they never change the frozen scientific identity of a task.

## Frozen cross-module contracts

New architecture contracts live under `src/tarca/contracts`:

- `ArtifactRef`: content/schema identity with an optional placement path;
- `ScientificIdentity`: protocol, experiment, task, model, data, and seed;
- `TaskSpec`, `TaskResult`, `TaskAttemptRecord`, `ResourceRequest`, and
  `ExecutionContext`;
- `GateDecision` and `SealedAccessGrant`;
- `EffectSignature`, `TaskManifest`, `ExperimentSummary`, and
  `MonitoringSnapshot`.

They are immutable dataclasses with explicit validation. `ArtifactRef.identity_key`
does not include `relative_path`, so moving an artifact cannot silently change
its identity.

`COMPLETED_TASK_POLICY` is globally `NEVER_RERUN`.

## Capability separation

`ForecastPredictor` remains prediction-only. The new
`MechanisticModelAdapter` is a separate Protocol and is not made a superclass
or replacement for the grandfathered adapter. Concept extraction, high-level
concept intervention, activation intervention, effect signatures, localization,
robustness, metrics, and experiment compilation have separate boundaries.

Every future scientific entry point is either a Protocol or raises
`UnimplementedCapabilityError(UNIMPLEMENTED_CAPABILITY)`. No placeholder
`PASS`, fake scientific metric, or fake scientific gate is returned by a
production module.

## Data and artifact boundaries

Data APIs distinguish dataset specification, temporal split, train-only fit,
transform, dataset hashing, access scope, and window materialization. Sealed
materialization is authorization-gated and intentionally unimplemented here.

Artifact publication follows the contract:

`temporary -> write -> flush -> hash -> reload/schema validate -> atomic publish -> completion marker`.

Completed artifacts are append-only. Existing Stage2/E02d-r4 stores continue to
own their existing execution semantics; this layer does not replace them.

## Configuration identity

The architecture distinguishes four future configuration classes:

1. `ScientificSpec` — protocol/model/data/metric semantics;
2. `RuntimeSpec` — deterministic runtime and precision identity;
3. `ExecutionSpec` — worker/resource/retry mechanics;
4. `MonitorSpec` — read-only telemetry presentation.

Existing E02d-r4 unified configs are not rewritten by this freeze.

## Out of scope

No Stage 3 activation patching, PLOT/OT/DAS implementation, abstraction metric
algorithm, DRO solver, finance module, real-market causal claim, formal
scientific experiment, training, server connection, sealed data access, or
P9/P10 execution is performed by this architecture task.

## Evidence

The local logical artifact catalog is
`artifacts/architecture/TARCA_ARTIFACT_INDEX_V1.json`. It maps existing results
to their logical owner and authority status without moving or renaming them.
That catalog is local evidence and is intentionally excluded from the public
skeleton sync; the portable architecture authority is the contract and module
registry set under `configs/architecture/`.
