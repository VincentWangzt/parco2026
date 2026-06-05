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

## M1 — halo-padded MPI inner kernel (N+2 stride, drop jl/jr branches)
serial    = 3130.5  ms
mpi_np1   = 331.596 ms  (Δ -5.6%)
mpi_np2   = 168.91  ms  (Δ -5.4%)
mpi_np4   = 92.71   ms  (Δ -11.0%)  ← clears 10% gate
mpi_np8   = 82.1487 ms  (Δ -4.6%)
mpi_np16  = 41.2978 ms  (Δ -1.3%)
cuda      = 5.25532 ms  (untouched)
verify: PASS / PASS / PASS
decision: KEEP — clear inner-kernel win at mpi_np4; no regressions

---

## M2 — non-blocking Isend/Irecv halo with interior-row compute overlap
serial    = 3121.69 ms
mpi_np1   = 331.956 ms  (Δ +0.1%)
mpi_np4   = 95.169  ms  (Δ +2.7% regression)
mpi_np8   = 81.7163 ms  (Δ -0.5%)
mpi_np16  = 41.0128 ms  (Δ -0.7%)
cuda      = 5.34096 ms  (untouched)
verify: PASS / PASS / PASS
decision: DROP — overlap does not pay off when MPI is already on shared-memory
          (vader) BTL; comm is cheap, interior-row split adds cache pollution.

---

## M3 — hybrid MPI + OpenMP (#pragma omp parallel for over rows)
serial    = 3116.39 ms
mpi_np1   = 332.917 ms  (pure MPI, OMP_NUM_THREADS=1)
mpi_np16  = 41.3805 ms
mpi_p1_t16  = 43.2765 ms   (1 rank × 16 OMP threads — best hybrid)
mpi_p2_t8   = 44.0003 ms
mpi_p4_t4   = 44.8177 ms
mpi_p8_t2   = 46.216  ms
cuda      = 5.2536 ms
verify: PASS / PASS / PASS
decision: DROP — every hybrid config is SLOWER than pure mpi_np16.
          200 OMP fork/join cycles per run cost more than the pure-MPI
          shared-memory comm they replace. Headline mpi_np16 stays the best
          CPU configuration.
NOTE: must keep `OMP_NUM_THREADS=1` for pure-MPI variants because the binary
      is built with -fopenmp; we'll need this even after reverting if the
      revert keeps -fopenmp. Plan: revert both code change AND -fopenmp flag
      so the rest of the exploration goes back to a clean MPI build.

---

## C1 — CUDA __ldg read-only loads on input buffer
cuda      = 5.24064 ms (Δ -0.4% vs M1's 5.255)
verify: PASS / PASS / PASS
decision: DROP — sm_60 already auto-routes const __restrict__ pointer loads
          through the read-only cache; the explicit __ldg is a wash.

---

## C2 — halo-padded device buffers (N+2)x(N+2), drop kernel-side bounds checks
cuda      = 5.38188 ms (Δ +2.4% regression vs M1's 5.255)
verify: PASS / PASS / PASS
decision: DROP — adds an extra 4 bytes of padding on each row and a stride
          parameter; saved branches were already free in nvcc's predicated
          form. Plus an extra cudaMemcpy to clear d_next's outer ring.

---

## C3 — CUDA shared-memory tiling (BX=BY=16, halo loaded into __shared__)
cuda      = 8.25059 ms (Δ +57% regression vs M1's 5.255)
verify: PASS / PASS / PASS
decision: DROP — heavy regression. T4's L1/texture cache already absorbed
          the redundant 8-neighbour reads; halo-load branches at block
          edges hurt warp uniformity, and the __syncthreads() drains the
          short kernel.

---

## C5 — CUDA block-size sweep (winner: 32x4)
Local 11-run median sweep at N=1024, T=200 (medians, ms):
  block 32x4  : 3.17358   <- WINNER
  block 32x8  : 3.94748
  block 32x16 : 4.09407
  block 16x8  : 5.29995
  block 16x16 : 5.34059   (old default)
  block 16x32 : 5.55952
  block 8x32  : 9.59802
  block 8x16  : 9.63123
  block 8x8   : 9.65136
With 32x4 baked into run.slurm bench (21 runs):
  cuda      = 3.17643 ms (Δ -39.6% vs M1's 5.255, speedup vs serial 978x)
verify: PASS / PASS / PASS
decision: KEEP — clear and large win. 32 threads in x = a warp doing fully
          coalesced byte loads; 4 rows = 4 warps/block keeps occupancy good.

---
