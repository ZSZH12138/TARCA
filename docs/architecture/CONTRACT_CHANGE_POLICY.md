# Contract Change Policy (CCP)

## Versioning

The current architecture version is `1.0`.

- Patch: documentation, validator tightening that rejects already-invalid
  values, or non-semantic test/tooling changes.
- Minor: additive optional fields, new Protocols, new registry entries, or new
  fail-closed capabilities that preserve existing signatures.
- Major: removal/renaming, changed tensor shape/dtype/time semantics, changed
  task identity, changed metric/gate meaning, changed sealed-access behavior,
  or any change that can alter scientific conclusions.

## CCP record

Every contract-affecting change receives a `CCP-XXXX` record containing:

1. old and new contract/version;
2. reason and affected modules;
3. scientific semantic impact;
4. historical compatibility impact;
5. migration and rollback plan;
6. tests and API-baseline changes;
7. experiment/protocol impact and required receipt updates.

No old receipt, historical artifact, sealed recipe, seed registry, checkpoint,
or completed-task marker may be edited to make a new contract appear valid.

## Grandfathered compatibility

If a new abstraction conflicts with a public interface already used by the
repository, the existing interface is authoritative. Add a wrapper or adapter
and document the compatibility binding. Do not rewrite the existing scientific
code merely to make it look like the new skeleton.

## Scientific-impact review

The following are automatically scientific-impacting and require a minor or
major review even when implemented in runtime code: batch size, gradient
accumulation, learning rate, precision/AMP/TF32, deterministic settings,
checkpoint identity, data split, seed, metric definition, aggregation, gate
threshold, or completed-task rerun policy.
