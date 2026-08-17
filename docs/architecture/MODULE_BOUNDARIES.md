# TARCA Module Boundaries

## Logical dependency graph

```text
contracts
  -> data
  -> models
  -> concepts
  -> interventions
  -> effects
  -> localization
  -> robustness
  -> metrics
  -> experiments -> orchestration -> runtime

contracts -> governance -> artifacts
contracts -> artifacts
orchestration/runtime -> monitoring
contracts -> backends
```

The machine-readable version is
`configs/architecture/dependency_rules_v1.json`. Because the grandfathered
Stage 1 scope test forbids new top-level science packages, future skeleton
interfaces are physically located under `src/tarca/architecture/skeleton/`
and are logically registered under their target module names.

## Module responsibilities and public entry points

| Module | Input/output contract | Current status |
| --- | --- | --- |
| `contracts` | immutable types, Protocols, validators, identity/error enums | frozen authority; grandfathered plus new architecture contracts |
| `data` | `DatasetSpec`/`SplitSpec`/`AccessScope` -> `WindowBatch`/`ArtifactRef` | existing implementation preserved; new generic entry points fail closed |
| `models` | `WindowBatch` -> `ForecastDistribution`; optional mechanistic adapter | existing predictors preserved; capability boundary added |
| `concepts` | `WindowBatch` -> `ConceptBatch`/`LeakageAudit` | `src/tarca/architecture/skeleton/concepts.py`; skeleton only |
| `interventions` | concept/window/model contracts -> `InterventionPairSet`/`InterventionResult` | `src/tarca/architecture/skeleton/interventions.py`; no Stage 3 implementation |
| `effects` | paired forecasts -> `EffectSignature` | `src/tarca/architecture/skeleton/effects.py`; signature and validator only |
| `localization` | effect signature + state -> `LocalizationResult` | `src/tarca/architecture/skeleton/localization.py`; state machine only |
| `robustness` | train/validation/test `EnvironmentSpec` -> solver contract | `src/tarca/architecture/skeleton/robustness.py`; split contract only |
| `metrics` | forecast/effect/result -> `MetricRecord` | `src/tarca/architecture/skeleton/metrics.py`; pure-consumer Protocols only |
| `experiments` | `ExperimentSpec` -> immutable `TaskManifest` -> summary/gate | `src/tarca/architecture/skeleton/experiments.py`; compiler and gate are unimplemented |
| `training` | existing Stage 0/1/2 and E02d-r4 training/runtime code | grandfathered; not moved or rewritten |
| `governance` | grant, sealed boundary, gate, receipt semantics | `src/tarca/architecture/skeleton/governance.py`; materialization fails closed |
| `artifacts` | append-only publication, verification, typed load, resolution | `src/tarca/architecture/skeleton/artifacts.py`; Protocol only |
| `orchestration` | lease tasks and record attempts | `src/tarca/architecture/skeleton/orchestration.py`; scheduler Protocol only |
| `runtime` | resources, qualification, execution, reconciliation | `src/tarca/architecture/skeleton/runtime.py`; fail-closed entry points |
| `monitoring` | status/resources/telemetry/terminal-safe ETA | `src/tarca/architecture/skeleton/monitoring.py`; read-only snapshot and Protocol only |
| `backends` | OT/intervention/storage Protocols | `src/tarca/architecture/skeleton/backends.py`; third-party types cannot escape |

## Forbidden coupling

- Contracts must not import training, orchestration, runtime, server, or
  scientific implementations.
- Data, models, and concepts are parallel contract consumers. Models must not
  depend on a concrete data implementation, and concepts must not depend on a
  concrete model implementation.
- Monitoring must not import data, models, concepts, interventions, effects,
  localization, robustness, or metrics.
- Science must not depend on scheduler/server state.
- Backend-specific types must not appear in core contracts.
- Cross-module calls must use typed contracts/Protocols; only explicitly typed
  metadata mappings may cross a boundary as mappings.

## File placement policy

This registry is logical. Existing files remain in `src/tarca/training`,
`artifacts/stage2`, and `configs/stage2`. New architecture evidence belongs in
`docs/architecture`, `configs/architecture`, `artifacts/architecture`, and
`tests/architecture`.
