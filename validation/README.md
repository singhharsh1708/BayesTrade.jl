# Validation

Scripts that measure the system rather than assert about it. They live outside `test/` on
purpose: a test answers yes or no and belongs in CI, while these produce numbers that a person
reads and compares against the last run.

```bash
julia --project=validation -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=validation validation/baseline.jl
```

Results land in `validation/results/` as JSON, and the readable version of each run is written up
in `docs/`.

Its own environment, like `examples/`, because `Pkg.add` inside the package project rewrites
`Project.toml` and takes the extension wiring with it.

| Script | Section of the brief | Output |
|---|---|---|
| `baseline.jl` | 1, 22 | `docs/VALIDATION_BASELINE.md` |
| `calibration.jl` | 9, 10, 11 | `docs/VALIDATION_CALIBRATION.md` |
| `benchmarks.jl` | 14, 15, 16, 17 | `docs/VALIDATION_CALIBRATION.md` |
| `mutations.jl` | 21 | `docs/VALIDATION_REPORT.md` |

`mutations.jl` exits non-zero if any mutation survives or any anchor has moved, so it can be run
as a gate rather than read as a report. The verdict is in `docs/VALIDATION_REPORT.md`.
