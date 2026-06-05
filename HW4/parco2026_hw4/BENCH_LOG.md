# BENCH_LOG.md — append-only Case L (N=1024, T=200) median benchmarks

Each entry is recorded after running `bash run.slurm` (with `RUNS=21`) on this
node (16 vCPUs, NVIDIA T4) and reading the `==== Case L summary ====` block.

Format per entry:

```
## <commit-sha-short> — <one-line description>
mpi_np1   = X.XX  ms   (Δ vs prev = +Y.Y%)
mpi_np4   = X.XX  ms
mpi_np16  = X.XX  ms
cuda      = X.XX  ms
verify: PASS/PASS/PASS (S/M/L)
decision: keep / drop  — reason
```

A "drop" entry is followed by a revert commit on the next row.

---

## baseline (a137197 + bench.py deletion in HW3) — original code
serial    = 3132.74 ms
mpi_np1   = 351.159 ms
mpi_np2   = 178.642 ms
mpi_np4   = 104.182 ms
mpi_np8   = 86.0899 ms
mpi_np16  = 41.8253 ms
cuda      = 5.33978 ms
verify: PASS / PASS / PASS (S/M/L)
decision: baseline reference

---
