# Iteration benchmark log (scratch — REPORT.md is rewritten only at the end)

GPU (local, dev): Tesla T4 (CC 7.5, idle, CUDA 12.1)
Protocol: 1 warmup, 21 timed iterations, median of CUDA scan (kernels-only) time.

| Version | Description | N=8388608 median (ms) | vs v0 | N=1000003 median (ms) | Status |
|---|---|---:|---:|---:|---|
| v0 | Baseline (3-pass Blelloch, per-recursion malloc, incl_to_excl_kernel) | 1.800 | 1.00× | 0.394 | baseline |
| v1 | Pre-allocated workspace; folded incl→excl into add_block_offsets | 1.589 | 1.13× | 0.178 | **kept** (12% faster at L, 2.2× at M) |
| v2 | Bank-conflict-free shared mem (CONFLICT_FREE_OFFSET padding) | 1.063 | 1.69× | 0.135 | **kept** (33% faster vs v1; biggest single win) |
