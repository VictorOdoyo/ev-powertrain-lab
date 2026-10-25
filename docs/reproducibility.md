# Reproducible captures

Run a validated ExperimentConfig with case, resolution_scale (1–3) and unsigned 32-bit seed. Named SHA-256 random streams keep sensor and uncertainty draws reproducible without relying on Python's randomized hash.

The dispatcher freezes source revision and dirty status before the physical solve. It records package/runtime versions, request hash, and resolved parameter hash. Study arrays are finite numeric NPZ fields, never pickle objects. Complex arrays retain their complex dtype.

Reports include parameter, summary and provenance JSON/CSV, a scientific plot, HTML and a complete SHA-256 manifest. verify detects altered or missing artifacts. Catalog validation rejects duplicate cases and paths escaping the catalog.

Use:
```sh
evlab batch --data-only --output outputs/catalog
evlab render outputs/catalog
evlab verify outputs/catalog
evlab replay outputs/catalog/foc-step/fields.npz --output outputs/replay
```
Replay preserves original solve provenance. A published gallery requires every default case from one clean committed source revision.
