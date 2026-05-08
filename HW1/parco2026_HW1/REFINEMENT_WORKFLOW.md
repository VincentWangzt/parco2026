# Iterative Refinement Workflow for `histogram_fast.cpp`

## Purpose

This document defines a generic methodology for an agent (or human) to iteratively
optimize `histogram_fast.cpp`. It describes **HOW** to approach optimization — not
**WHAT** specific optimizations to try.

---

## 1. Invariants (What CANNOT Change)

| Constraint | Reason |
|------------|--------|
| Correctness | Must pass `python3 check.py <input> <output>` |
| Input file format | Compatibility with `gen_input.py` |
| Output file format | Compatibility with `check.py` (M integer lines) |
| CLI interface | `./histogram_fast <input> <output>` |
| Timing protocol | `omp_get_wtime()`, 2 warmup + 5 measured runs, mean±std |
| Max threads | 16 (honor `OMP_NUM_THREADS`) |
| Language | C/C++ with OpenMP |

---

## 2. Scope (What CAN Change)

Anything in `histogram_fast.cpp` and its compilation flags, including but not limited to:

- Algorithmic approach (loop structure, multi-pass, tiling)
- Memory layout (alignment, padding, allocation strategy)
- Compiler hints (`__builtin_prefetch`, `__builtin_expect`, `__restrict__`)
- OpenMP directives (schedule policy, collapse, nowait, task, SIMD)
- Preprocessor defines (CACHE_LINE, unroll factors, batch sizes)
- Compiler flags in Makefile (`-O` level, `-march`, `-mtune`, `-ffast-math`, LTO)
- Data types (if exact correctness is preserved)
- SIMD intrinsics (SSE/AVX manual vectorization)
- Thread affinity hints (`OMP_PROC_BIND`, `OMP_PLACES`)

---

## 3. Measurement Protocol

### Setup
```bash
make histogram_fast
export OMP_NUM_THREADS=16
```

### Primary benchmark (for decision making)
```bash
./histogram_fast input2.dat output_fast.dat
# Read TIMING line from stderr: TIMING,fast,16,100000000,256,<mean>,<std>
```

Use `input2.dat` (100M points) — it's large enough for stable measurements.

### Correctness check (fast, on small input)
```bash
./histogram_fast input1.dat output_fast.dat
python3 check.py input1.dat output_fast.dat
```

### Additional validation (after keeping a change)
```bash
# Also verify on large input
python3 check.py input2.dat output_fast.dat
```

---

## 4. Refinement Cycle

Repeat until convergence:

### Step 1: Record Baseline
```
BASELINE_MEAN = <current mean time>
BASELINE_STD  = <current std>
```

### Step 2: Hypothesize
Before making any change, write down:
- What you're changing
- Why you expect it to help
- What mechanism it exploits (cache, ILP, vectorization, etc.)

### Step 3: Implement
Make **one atomic change**. Do not combine multiple unrelated optimizations —
you won't know which one helped (or hurt).

### Step 4: Verify Correctness
```bash
make histogram_fast
./histogram_fast input1.dat output_fast.dat > /dev/null
python3 check.py input1.dat output_fast.dat
```

**If correctness fails → REVERT immediately. Do not proceed.**

### Step 5: Measure
```bash
export OMP_NUM_THREADS=16
./histogram_fast input2.dat output_fast.dat
```

Record `NEW_MEAN` and `NEW_STD`.

### Step 6: Decide

```
improvement = (BASELINE_MEAN - NEW_MEAN) / BASELINE_MEAN

IF NEW_MEAN + NEW_STD < BASELINE_MEAN - BASELINE_STD:
    # Clear improvement (distributions don't overlap)
    → KEEP. Update baseline.

ELIF NEW_MEAN - NEW_STD > BASELINE_MEAN + BASELINE_STD:
    # Clear regression
    → REVERT.

ELSE:
    # Ambiguous — within noise
    # Run 3 additional measurement cycles to confirm
    # If still ambiguous: KEEP only if change reduces code complexity
    # Otherwise: REVERT (prefer simplicity)
```

### Step 7: Log
Append to the optimization log (in the source file or a separate file):
```
v<N>: <description> — <old_mean>s → <new_mean>s (<±X.X%>) [kept/reverted]
```

---

## 5. Convergence Criteria

Stop iterating when ANY of these conditions hold:

1. **Noise floor reached:** Last 3 consecutive attempts all fall within measurement noise
   (mean differences smaller than 1 std of either measurement)

2. **Bandwidth bound reached:** The computation time scales linearly with N and is within
   2× of the theoretical minimum (N × 8 bytes / memory bandwidth)

3. **Diminishing returns:** Total time spent optimizing exceeds the value of further gains

---

## 6. Debugging Performance

If an optimization doesn't help when expected:

1. **Check if memory-bound:** If time scales with N (not compute), ALU optimizations won't help
2. **Check thread scaling:** Run at 1, 4, 8, 16 threads — if 1→4 is good but 8→16 plateaus, you're bandwidth-limited
3. **Check compiler output:** `g++ -S -O3 ... histogram_fast.cpp` — is the compiler already doing what you tried?
4. **Profile:** `perf stat ./histogram_fast input2.dat /dev/null` for IPC, cache misses, branch mispredictions

---

## 7. Common Pitfalls

- **Don't optimize I/O** — it's not timed
- **Don't assume thread count** — code must work for any OMP_NUM_THREADS ≤ 16
- **Don't use `-ffast-math` without verifying** — it can change floating-point results
- **Don't hardcode M=256** — the code must work for any M from the input file
- **Don't ignore 1-thread performance** — the serial path should also be fast
- **Measure consistently** — same machine state, no other heavy processes running
