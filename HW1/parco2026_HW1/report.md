# PARCO 2026 HW1: OpenMP Histogram — Performance Report

## 1. Problem Description

Given N floating-point numbers in [min, max] and M bins, compute a histogram counting data points per bin. The serial algorithm is O(N): for each data point, compute bin index and increment counter.

**Parallelization challenges:**
- **Race condition:** Multiple threads executing `hist[bin]++` simultaneously lose counts (read-modify-write is non-atomic)
- **False sharing:** Adjacent thread-local histograms sharing cache lines cause coherence invalidations

---

## 2. Implementation Strategies

### 2.1 Serial Baseline
Direct implementation of the algorithm. Single-threaded, no synchronization needed.

### 2.2 Atomic (`#pragma omp atomic`)
Each `hist[bin]++` is protected by a hardware atomic increment. Simple but creates massive contention with M=256 bins — with 16 threads and uniform data, each bin is contested with probability ≈1/256 per access.

### 2.3 Critical Section (`#pragma omp critical`)
The entire increment is wrapped in a mutex lock. ALL operations serialize regardless of which bin they access — catastrophic performance with multiple threads.

### 2.4 Thread-Private + Manual Reduction (no padding)
Each thread maintains a private histogram array, then results are merged with a serial reduction loop. No contention during the parallel phase, but contiguous allocation means thread-boundary cache lines may be shared (false sharing when M×4 is not a multiple of 64).

### 2.5 Thread-Private + Padded + Manual Reduction
Same as above but each thread's array is padded to cache-line boundaries (stride = ceil(M/16)×16 + 16 ints). Eliminates false sharing.

### 2.6 OpenMP Array Reduction (compiler-managed)
Uses OpenMP 4.5+ `reduction(+:arr[:M])` clause. The compiler allocates private copies and handles the merge internally. Functionally equivalent to manual private+reduction but potentially better optimized by the runtime.

### 2.7 Optimized Fast Version
Thread-private + padded + aligned allocation (`posix_memalign`) + reciprocal multiply (avoid division in inner loop) + parallel merge phase + `-O3 -march=native`.

---

## 3. Experimental Environment

| Parameter | Value |
|-----------|-------|
| Data sizes | N=10^7 (10M), N=10^8 (100M) |
| Bins | M=256 |
| Value range | [0.0, 1.0) uniform |
| Thread counts | 1, 2, 4, 8, 16 |
| Timing | compute only (I/O excluded), 2 warmup + 5 measured runs, mean±std |
| Compiler | g++ -std=c++11 |
| Optimization | -O2 (strategy comparison), -O3 -march=native (fast version) |

---

## 4. Performance Results

### 4.1 Input1.dat (N=10M, M=256)

| Strategy | 1T (s) | 2T (s) | 4T (s) | 8T (s) | 16T (s) |
|----------|--------|--------|--------|--------|---------|
| serial | 0.0165 | — | — | — | — |
| atomic | 0.0326 | 0.2199 | 0.2418 | 0.1744 | 0.1602 |
| critical | 0.0642 | 0.3564 | 0.5032 | 1.8250 | 2.4082 |
| private | 0.0168 | 0.0137 | 0.0072 | 0.0054 | 0.0029 |
| padded | 0.0169 | 0.0088 | 0.0045 | 0.0024 | 0.0021 |
| reduction | 0.0169 | 0.0088 | 0.0045 | 0.0024 | 0.0015 |
| **fast** | 0.0118 | 0.0064 | 0.0034 | 0.0018 | 0.0019 |

### 4.2 Input2.dat (N=100M, M=256)

| Strategy | 1T (s) | 2T (s) | 4T (s) | 8T (s) | 16T (s) |
|----------|--------|--------|--------|--------|---------|
| serial | 0.1700 | — | — | — | — |
| atomic | 0.3236 | 2.2685 | 1.9271 | 2.0345 | 1.6407 |
| critical | 0.6766 | 3.6512 | 6.9145 | 14.973 | 25.034 |
| private | 0.1682 | 0.1379 | 0.0880 | 0.0500 | 0.0292 |
| padded | 0.1697 | 0.0900 | 0.0448 | 0.0245 | 0.0129 |
| reduction | 0.1702 | 0.0848 | 0.0441 | 0.0238 | 0.0124 |
| **fast** | 0.1180 | 0.0646 | 0.0312 | 0.0157 | 0.0119 |

### 4.3 Speedup (vs serial, input2.dat N=100M)

| Strategy | 1T | 2T | 4T | 8T | 16T |
|----------|-----|------|------|-------|-------|
| atomic | 0.53× | 0.07× | 0.09× | 0.08× | 0.10× |
| critical | 0.25× | 0.05× | 0.02× | 0.01× | 0.01× |
| private | 1.01× | 1.23× | 1.93× | 3.40× | 5.81× |
| padded | 1.00× | 1.89× | 3.80× | 6.95× | **13.15×** |
| reduction | 1.00× | 2.00× | 3.86× | 7.13× | **13.69×** |
| **fast** | 1.44× | 2.63× | 5.45× | 10.80× | **14.24×** |

---

## 5. Analysis

### 5.1 Why Atomic and Critical Are Catastrophic

**Critical section** serializes ALL increments behind a single mutex — regardless of whether threads access the same bin. With 16 threads, it's 147× slower than serial (25.0s vs 0.17s). Each thread spends nearly all its time waiting for the lock.

**Atomic operations** are better in principle (only serialize conflicting accesses), but with M=256 bins and uniform data, the cache-line-level contention is still severe. The x86 `lock xadd` instruction forces cache-line ownership transfer on every atomic increment, even when different threads access different bins that happen to share a cache line (256 ints = 1024 bytes = 16 cache lines; 16 threads frequently contend on the same lines).

Both strategies get **slower** with more threads — negative scaling.

### 5.2 Private vs Padded: False Sharing Impact

With M=256, each thread-local array is 1024 bytes = exactly 16 cache lines. Since 256×4 = 1024 is a multiple of 64, the arrays are naturally cache-line-aligned in contiguous allocation. Yet `padded` still outperforms `private` by 2.3× at 16 threads on input2.dat (0.0129s vs 0.0292s).

**Why?** The `std::vector` allocator doesn't guarantee that the starting address of `local_hists` is cache-line-aligned. If the vector's internal buffer starts at a non-64-byte-aligned address, ALL thread boundaries become misaligned. The padded version adds an extra 64-byte gap between threads, making it immune to the base address offset.

### 5.3 Reduction vs Padded

OpenMP's `reduction(+:arr[:M])` performs slightly better than our manual padded version (13.69× vs 13.15×). The compiler likely uses aligned allocation internally and may optimize the merge phase (possibly parallelized or SIMD-vectorized).

### 5.4 Fast Version Advantage

The fast version achieves 14.24× speedup at 16 threads — the best result. Key advantages over the `reduction` strategy:
- **Reciprocal multiply:** Avoids the costly floating-point division per data point
- **Aligned allocation:** `posix_memalign` guarantees 64-byte alignment (no vector overhead)
- **Parallel merge:** The reduction of 16 thread-local arrays is parallelized across bins
- **-O3 -march=native:** Enables aggressive compiler optimizations

### 5.5 Scaling Behavior and Bandwidth Limit

For the fast version on input2.dat:
- 1→2 threads: 1.83× (near ideal)
- 2→4 threads: 1.77× (good)
- 4→8 threads: 1.98× (near ideal)
- 8→16 threads: 1.32× (diminishing — approaching memory bandwidth ceiling)

The theoretical minimum time is limited by memory bandwidth. Reading 100M doubles = 800MB from DRAM. At typical memory bandwidth of ~40-50 GB/s, the floor is ~16-20ms. Our best result (11.9ms at 16T) suggests we're already close to or slightly exceeding the bandwidth ceiling (possibly benefiting from L3 cache effects).

---

## 6. False Sharing Experiment (Extension Topic)

### 6.1 Theoretical Model

For a thread-private histogram of M ints allocated contiguously:
- Array size per thread: M × 4 bytes
- **Spill bytes** = (M × 4) mod 64
- If spill > 0: adjacent threads share a cache line at their boundary
- False sharing causes cache-line invalidations on every write to the shared line

**Prediction:** M values that are multiples of 16 (i.e., M×4 is a multiple of 64) will NOT exhibit false sharing. Non-multiples WILL.

### 6.2 Experiment A: Dense M Sweep (16 threads, padding=0)

| M | Array (B) | Spill (B) | Aligned? | Time (ms) | vs aligned avg |
|---|-----------|-----------|----------|-----------|----------------|
| 8 | 32 | 32 | No | 3.72 | — |
| 9 | 36 | 36 | No | 6.60 | — |
| 12 | 48 | 48 | No | 4.76 | — |
| 16 | 64 | 0 | **Yes** | 7.72 | baseline |
| 17 | 68 | 4 | No | 7.46 | — |
| 20 | 80 | 16 | No | 4.84 | — |
| 24 | 96 | 32 | No | 3.07 | — |
| 32 | 128 | 0 | **Yes** | 1.52 | baseline |
| 33 | 132 | 4 | No | 5.08 | +234% |
| 36 | 144 | 16 | No | 4.49 | +195% |
| 48 | 192 | 0 | **Yes** | 2.01 | baseline |
| 64 | 256 | 0 | **Yes** | 1.30 | baseline |
| 65 | 260 | 4 | No | 5.45 | +321% |
| 72 | 288 | 32 | No | 3.03 | +134% |
| 96 | 384 | 0 | **Yes** | 1.54 | baseline |
| 128 | 512 | 0 | **Yes** | 3.25 | baseline |
| 129 | 516 | 4 | No | 3.15 | -3% |
| 256 | 1024 | 0 | **Yes** | 2.38 | baseline |

### 6.3 Paired Analysis (Experiment B)

| Pair | Aligned | Misaligned | Penalty | Observation |
|------|---------|------------|---------|-------------|
| M=8 vs M=9 | 3.72ms | 6.60ms | **+78%** | Small array, high contention density |
| M=16 vs M=17 | 7.72ms | 7.46ms | -3% | Surprising: M=16 is slow (see analysis) |
| M=32 vs M=33 | 1.52ms | 5.08ms | **+234%** | Clear false sharing |
| M=32 vs M=36 | 1.52ms | 4.49ms | **+195%** | Larger spill, same effect |
| M=64 vs M=65 | 1.30ms | 5.45ms | **+321%** | Maximum observed penalty |
| M=64 vs M=72 | 1.30ms | 3.03ms | **+134%** | Still significant |

### 6.4 Key Observations

1. **False sharing penalty is massive** for moderate M: M=64→65 causes a 4.2× slowdown
2. **The penalty is NOT proportional to spill amount**: spill=4B (pair 3) and spill=16B (pair 4) produce similar effects. Any spill = full cache line invalidation
3. **M=16 anomaly:** Despite being aligned, M=16 (64 bytes = 1 cache line per thread) is abnormally slow. This is because the ENTIRE histogram fits in one cache line — causing true sharing contention on the single line, not false sharing
4. **Large M (128, 256):** Penalties decrease because the conflicting boundary bins are accessed less frequently relative to total work

### 6.5 Experiment C: Thread Scaling

| Threads | M=16 (ms) | M=17 (ms) | Ratio |
|---------|-----------|-----------|-------|
| 1 | 16.57 | 16.29 | 0.98× |
| 2 | 23.54 | 23.39 | 0.99× |
| 4 | 16.87 | 14.13 | 0.84× |
| 8 | 9.63 | 7.13 | 0.74× |
| 16 | 6.43 | 6.24 | 0.97× |

**Observation:** For these very small M values (16-17), the performance is dominated by other effects (true sharing on the small histogram itself, thread overhead). The false sharing signal is clearest at moderate M (32-96) where the histogram is large enough to avoid true sharing but small enough that boundary contention is a significant fraction of work.

### 6.6 Experiment D: Padding Threshold (M=17, 16 threads)

| Padding (B) | Time (ms) | vs pad=0 |
|-------------|-----------|----------|
| 0 | 7.54 | baseline |
| 4 | 5.24 | -30% |
| 8 | 5.17 | -31% |
| 16 | 4.65 | -38% |
| 32 | 2.92 | -61% |
| **64** | **1.23** | **-84%** |
| 128 | 1.31 | -83% |

**Critical threshold: 64 bytes (one cache line).** Performance improves gradually with partial padding but jumps dramatically at 64B — confirming that the issue is cache-line sharing, and the fix requires separating threads by at least one full line.

### 6.7 Memory Footprint Analysis

Extra memory per thread from padding:
- Padding = 64B → 16 threads × 64B = **1024 bytes total overhead**
- For M=256: base memory = 16 × 1024B = 16KB; padding adds 6.25%
- For M=17: base memory = 16 × 68B = 1088B; padding adds 94% — but absolute cost is still only 1KB

**Conclusion:** The memory cost of cache-line padding is negligible (1KB for 16 threads) while the performance benefit can be 4× or more. Always pad.

### 6.8 Prediction vs Measurement

| Prediction | Observed | Match? |
|-----------|----------|--------|
| Aligned M → no false sharing | Mostly confirmed (M=32,48,64,96 are fast) | ✓ |
| Misaligned M → false sharing | Confirmed for M=33,65 (+234%, +321%) | ✓ |
| Penalty ∝ spill amount | NOT confirmed — any spill ≈ same penalty | ✗ (penalty is binary, not proportional) |
| M=16 should be fast (aligned) | NOT fast — true sharing dominates | ✗ (model incomplete for very small M) |
| M=128,256 should show penalty if misaligned | M=129 vs 128: only -3% difference | Partially ✗ (penalty decreases with M) |
| Penalty grows with thread count | Not clearly observed for small M | Partially ✗ |

**Model limitations:** The simple "spill = false sharing" model works well for moderate M (32-96) but breaks down at extremes:
- Very small M (≤16): true sharing dominates
- Very large M (≥128): boundary bins are too infrequently accessed to matter

---

## 7. Conclusion

### Best Strategy
**OpenMP array reduction** or **padded thread-private + manual reduction** achieve the best performance with simple code. The `fast` version provides marginal additional improvement through micro-optimizations.

### Maximum Achieved Speedup
- **14.24×** at 16 threads on 100M data points (fast version)
- **13.69×** at 16 threads (reduction, no micro-optimizations needed)

### Key Insights
1. Histogram computation is **memory-bandwidth-bound** for large N — once race conditions are eliminated, performance is limited by DRAM throughput
2. **Atomic/critical are anti-patterns** for histogram — they cause negative scaling (slower with more threads)
3. **False sharing is significant** for moderate M (32-96) with 4× penalties, but irrelevant for M=256 (naturally aligned)
4. **Cache-line padding** (64 bytes between thread arrays) is a universal, cheap fix for false sharing
5. The OpenMP `reduction` clause is competitive with hand-tuned code — compiler does a good job

### Practical Recommendation
For production histogram code: use `reduction(+:arr[:M])` if your compiler supports OpenMP 4.5+ array reduction. Otherwise, use thread-private arrays with cache-line padding. Always compile with `-O2` or higher.
