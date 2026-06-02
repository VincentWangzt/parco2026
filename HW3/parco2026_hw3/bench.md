# Iteration benchmark log (scratch — REPORT.md is rewritten only at the end)

GPU (local, dev): Tesla T4 (CC 7.5, idle, CUDA 12.1)
Protocol: 1 warmup, 21 timed iterations, median of CUDA scan (kernels-only) time.

| Version | Description | N=8388608 median (ms) | vs v0 | N=1000003 median (ms) | Status |
|---|---|---:|---:|---:|---|
| v0 | Baseline (3-pass Blelloch, per-recursion malloc, incl_to_excl_kernel) | 1.800 | 1.00× | 0.394 | baseline |
| v1 | Pre-allocated workspace; folded incl→excl into add_block_offsets | 1.589 | 1.13× | 0.178 | **kept** (12% faster at L, 2.2× at M) |
| v2 | Bank-conflict-free shared mem (CONFLICT_FREE_OFFSET padding) | 1.063 | 1.69× | 0.135 | **kept** (33% faster vs v1; biggest single win) |
| v3a (tried, reverted) | int32 input + int2 vectorized loads, consecutive-pair shared-mem layout | 1.100 | 1.64× | 0.138 | **reverted** — int2 forced a 2-way bank-conflict layout that more than ate the load savings |
| v3 | int32 input only (interleaved layout retained from v2; scalar reads, long long sdata) | 1.018 | 1.77× | 0.134 | **kept** — ~4% faster vs v2 in 51-iter head-to-head; halves d_x footprint (32 MB instead of 64 MB at L) |
| v4 | Hand-rolled itoa + single fwrite (post-timed-scan only) | 0.893 (drift) | n/a | 0.133 | **kept** — pure I/O QOL: end-to-end wall time at L drops 3.5 s → 0.71 s. Scan-only number drifted with steady-state GPU clocks during this measurement; do not attribute the ~14% delta to v4. |
