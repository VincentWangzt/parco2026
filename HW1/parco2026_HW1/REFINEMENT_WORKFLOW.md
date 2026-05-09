# Iterative Refinement Workflow for `histogram_fast.cpp`

## Purpose

This document defines a generic methodology for an agent (or human) to iteratively
optimize `histogram_fast.cpp`. It describes **HOW** to approach optimization — not
**WHAT** specific optimizations to try.

---

## 1. Invariants (What CANNOT Change)

| Constraint | Reason |
|------------|--------|
| Correctness | Must produce byte-identical output to `./serial` (or pass `python3 check.py`) |
| Binary I/O | Input: `.bin` files; Output: `.bin` files |
| CLI interface | `./histogram_fast <input.bin> <output.bin>` |
| Timing protocol | `omp_get_wtime()`, 2 warmup + 5 measured runs, mean±std to stderr |
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

## 3. Directory Layout

All intermediate outputs go in `results_test/`. Create it if it doesn't exist.

```
results_test/
├── out_serial_input1.bin      # serial reference output (input1)
├── out_serial_input2.bin      # serial reference output (input2)
├── out_fast_input1.bin        # histogram_fast output (input1)
├── out_fast_input2.bin        # histogram_fast output (input2)
└── refinement.log             # optimization log (see §7)
```

---

## 4. Measurement Protocol

### Setup
```bash
make histogram_fast serial
export OMP_NUM_THREADS=16
mkdir -p results_test
```

### Generate binary inputs (if missing)
```bash
# Generate text inputs first if needed
python3 gen_input.py
# Convert to binary
python3 convert_to_binary.py input1.dat input1.bin
python3 convert_to_binary.py input2.dat input2.bin
```

### Generate serial reference outputs (once, or after input regen)
```bash
./serial input1.bin results_test/out_serial_input1.bin
./serial input2.bin results_test/out_serial_input2.bin
```

### Primary benchmark (for decision making)
```bash
./histogram_fast input2.bin results_test/out_fast_input2.bin 20
# Read TIMING line from stderr: TIMING,fast,16,100000000,256,<mean>,<std>
```

Use `input2.bin` (100M points) — it's large enough for stable measurements.

### Correctness check (two-tier: fast binary diff, fallback to check.py)
```bash
./histogram_fast input1.bin results_test/out_fast_input1.bin

# Tier 1: byte-identical to serial reference (fast, preferred)
if cmp -s results_test/out_fast_input1.bin results_test/out_serial_input1.bin; then
    echo "CORRECT (binary match)"
else
    # Tier 2: fall back to check.py via text conversion
    python3 convert_to_binary.py --reverse results_test/out_fast_input1.bin /tmp/out_fast_check.dat
    python3 check.py input1.dat /tmp/out_fast_check.dat
    rm -f /tmp/out_fast_check.dat
fi
```

### Additional validation (after keeping a change)
```bash
# Also verify on large input
./histogram_fast input2.bin results_test/out_fast_input2.bin 20
# Tier 1: byte-identical to serial reference (fast, preferred)
if cmp -s results_test/out_fast_input2.bin results_test/out_serial_input2.bin; then
    echo "CORRECT (binary match)"
else
    # Tier 2: fall back to check.py via text conversion
    python3 convert_to_binary.py --reverse results_test/out_fast_input2.bin /tmp/out_fast_check.dat
    python3 check.py input2.dat /tmp/out_fast_check.dat
    rm -f /tmp/out_fast_check.dat
fi
```

---

## 5. Refinement Cycle

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
./histogram_fast input1.bin results_test/out_fast_input1.bin 2>/dev/null
cmp -s results_test/out_fast_input1.bin results_test/out_serial_input1.bin
```

If `cmp` reports a mismatch but you suspect a legitimate tie-breaking difference,
fall back to check.py:
```bash
python3 convert_to_binary.py --reverse results_test/out_fast_input1.bin /tmp/out_fast_check.dat
python3 check.py input1.dat /tmp/out_fast_check.dat
rm -f /tmp/out_fast_check.dat
```

**If correctness fails → REVERT immediately. Do not proceed.**

### Step 5: Measure
```bash
export OMP_NUM_THREADS=16
./histogram_fast input2.bin results_test/out_fast_input2.bin 20
```

Record `NEW_MEAN` and `NEW_STD` from the TIMING line on stderr.

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
Append to `results_test/refinement.log`:
```
[YYYY-MM-DD HH:MM] v<N>: <description> — <old_mean>s → <new_mean>s (<±X.X%>) [kept/reverted]
  Hypothesis: <why you expected it to help>
  Mechanism: <cache/ILP/vectorization/datatype-cast/etc.>
  Notes: <any observations, e.g. "compiler was already doing this">
```

---

## 6. Debugging Performance

If an optimization doesn't help when expected:

1. **Check if memory-bound:** If time scales with N (not compute), ALU optimizations won't help
2. **Check compiler output:** `g++ -S -O3 ... histogram_fast.cpp` — is the compiler already doing what you tried?
3. **Profile:** `perf stat ./histogram_fast input2.bin /dev/null` for IPC, cache misses, branch mispredictions

---

## 7. Common Pitfalls

- **Don't assume thread count** — code must work for any OMP_NUM_THREADS ≤ 16
- **Don't use `-ffast-math` without verifying** — it can change floating-point results
- **Don't hardcode M=256** — the code must work for any M from the input file
- **Don't ignore 1-thread performance** — the serial path should also be fast
- **Measure consistently** — same machine state, no other heavy processes running
- **Always use binary I/O** — text I/O is only for check.py fallback validation

---

## 8. Quick-Reference Commands

```bash
# Full cycle (build, check, bench) — copy-paste ready:
make histogram_fast \
  && ./histogram_fast input1.bin results_test/out_fast_input1.bin 2>/dev/null \
  && cmp -s results_test/out_fast_input1.bin results_test/out_serial_input1.bin \
  && echo "✓ Correct" \
  && OMP_NUM_THREADS=16 ./histogram_fast input2.bin results_test/out_fast_input2.bin 20

# If cmp fails, debug with check.py:
python3 convert_to_binary.py --reverse results_test/out_fast_input1.bin /tmp/_check.dat \
  && python3 check.py input1.dat /tmp/_check.dat \
  && rm -f /tmp/_check.dat
```
