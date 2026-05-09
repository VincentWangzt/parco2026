#!/bin/bash
# benchmark.sh — Full strategy comparison sweep for PARCO HW1
# Sweeps: strategies × thread counts × input files
# Output: results/benchmark_TIMESTAMP.csv + summary table
#
# Usage: ./benchmark.sh [output_dir]

set -euo pipefail

RESULT_DIR="${1:-results}"
mkdir -p "$RESULT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV="$RESULT_DIR/benchmark_${TIMESTAMP}.csv"

# ── Configuration ─────────────────────────────────────────────────────────────
# Strategies are auto-detected from the compiled binary after build.
# (reduction is included automatically when OpenMP 4.5+ is available)
THREAD_COUNTS="1 2 4 8 16"
INPUTS="input1.dat input2.dat"
NUM_RUNS=20
WARMUP=2
# ──────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════"
echo " PARCO HW1 Benchmark Sweep"
echo " Output: $CSV"
echo "═══════════════════════════════════════════════════"
echo ""

# Build
echo "[BUILD] Compiling all targets..."
make all 2>&1 | grep -v "^make\[" || true
echo ""

# Auto-detect available strategies from the compiled binary
# The binary prints available strategy names in its usage line when invoked without args
STRATEGIES=$(./histogram_omp 2>&1 | grep -oP '(?<=Strategies:).*' | tr -s ' ' | xargs)
if [ -z "$STRATEGIES" ]; then
    # Fallback: safe list that works with any OpenMP version
    STRATEGIES="serial atomic critical private padded"
fi
echo "[INFO] Available strategies: $STRATEGIES"
echo ""

# Generate inputs if missing
for inp in $INPUTS; do
    if [ ! -f "$inp" ]; then
        echo "[GEN] Generating $inp..."
        python3 gen_input.py
        break
    fi
done

# Convert inputs to binary for faster I/O
for inp in $INPUTS; do
    bin_inp="${inp%.dat}.bin"
    if [ ! -f "$bin_inp" ]; then
        echo "[CONVERT] $inp -> $bin_inp"
        python3 convert_to_binary.py "$inp" "$bin_inp"
    fi
done

# CSV header
echo "strategy,threads,N,M,mean_sec,std_sec,correct" > "$CSV"

# ── Collect serial baseline ──────────────────────────────────────────────────
echo "[SERIAL BASELINE]"
for input in $INPUTS; do
    echo -n "  serial / $input: "
    bin_input="${input%.dat}.bin"
    serial_out="$RESULT_DIR/out_serial_$(basename $input .dat).bin"
    timing_line=$(./serial "$bin_input" "$serial_out" 2>&1 | grep "^TIMING,")
    mean=$(echo "$timing_line" | cut -d',' -f6)
    std=$(echo "$timing_line" | cut -d',' -f7)
    N_val=$(echo "$timing_line" | cut -d',' -f4)
    M_val=$(echo "$timing_line" | cut -d',' -f5)

    # Verify via check.py (convert to text only here, for the baseline)
    serial_out_txt="$RESULT_DIR/out_serial_$(basename $input .dat).dat"
    python3 convert_to_binary.py --reverse "$serial_out" "$serial_out_txt"
    correct="yes"
    if ! python3 check.py "$input" "$serial_out_txt" 2>&1 | grep -q "CORRECT"; then
        correct="no"
    fi
    rm -f "$serial_out_txt"
    echo "mean=${mean}s std=${std}s correct=${correct}"
    echo "serial,1,${N_val},${M_val},${mean},${std},${correct}" >> "$CSV"
done
echo ""

# ── Strategy sweeps ──────────────────────────────────────────────────────────
for input in $INPUTS; do
    echo "[SWEEP] $input"
    bin_input="${input%.dat}.bin"
    for strategy in $STRATEGIES; do
        for threads in $THREAD_COUNTS; do
            # Serial strategy only runs with 1 thread (already done above from serial binary)
            if [ "$strategy" = "serial" ]; then
                if [ "$threads" != "1" ]; then
                    continue
                fi
                # Run via histogram_omp serial for consistency check
            fi

            # Cap thread count for atomic/critical (they scale negatively)
            if [ "$strategy" = "atomic" ] || [ "$strategy" = "critical" ]; then
                if [ "$threads" -gt 4 ]; then
                    continue
                fi
            fi

            export OMP_NUM_THREADS=$threads
            output_file="$RESULT_DIR/out_${strategy}_t${threads}_$(basename $input .dat).bin"

            echo -n "  $strategy / t=$threads: "

            timing_line=$(./histogram_omp "$strategy" "$bin_input" "$output_file" $NUM_RUNS $WARMUP 2>&1 | grep "^TIMING,")
            mean=$(echo "$timing_line" | cut -d',' -f6)
            std=$(echo "$timing_line" | cut -d',' -f7)
            N_val=$(echo "$timing_line" | cut -d',' -f4)
            M_val=$(echo "$timing_line" | cut -d',' -f5)

            # Verify correctness: compare binary outputs directly, only invoke check.py on mismatch
            serial_ref="$RESULT_DIR/out_serial_$(basename $input .dat).bin"
            correct="yes"
            if cmp -s "$output_file" "$serial_ref"; then
                :  # byte-identical to verified serial output
            else
                # Convert to text and run check.py
                output_file_txt="${output_file%.bin}.dat"
                python3 convert_to_binary.py --reverse "$output_file" "$output_file_txt"
                if ! python3 check.py "$input" "$output_file_txt" 2>&1 | grep -q "CORRECT"; then
                    correct="no"
                fi
                rm -f "$output_file_txt"
            fi

            echo "mean=${mean}s std=${std}s correct=${correct}"
            echo "${strategy},${threads},${N_val},${M_val},${mean},${std},${correct}" >> "$CSV"
        done
    done

    # Fast version sweep
    echo "  [FAST]"
    for threads in $THREAD_COUNTS; do
        export OMP_NUM_THREADS=$threads
        output_file="$RESULT_DIR/out_fast_t${threads}_$(basename $input .dat).bin"

        echo -n "  fast / t=$threads: "

        timing_line=$(./histogram_fast "$bin_input" "$output_file" 2>&1 | grep "^TIMING,")
        mean=$(echo "$timing_line" | cut -d',' -f6)
        std=$(echo "$timing_line" | cut -d',' -f7)
        N_val=$(echo "$timing_line" | cut -d',' -f4)
        M_val=$(echo "$timing_line" | cut -d',' -f5)

        # Verify correctness: compare binary outputs directly, only invoke check.py on mismatch
        serial_ref="$RESULT_DIR/out_serial_$(basename $input .dat).bin"
        correct="yes"
        if cmp -s "$output_file" "$serial_ref"; then
            :  # byte-identical to verified serial output
        else
            # Convert to text and run check.py
            output_file_txt="${output_file%.bin}.dat"
            python3 convert_to_binary.py --reverse "$output_file" "$output_file_txt"
            if ! python3 check.py "$input" "$output_file_txt" 2>&1 | grep -q "CORRECT"; then
                correct="no"
            fi
            rm -f "$output_file_txt"
        fi

        echo "mean=${mean}s std=${std}s correct=${correct}"
        echo "fast,${threads},${N_val},${M_val},${mean},${std},${correct}" >> "$CSV"
    done

    # Fast division version sweep (standard division, no reciprocal)
    echo "  [FAST_DIV]"
    for threads in $THREAD_COUNTS; do
        export OMP_NUM_THREADS=$threads
        output_file="$RESULT_DIR/out_fast_div_t${threads}_$(basename $input .dat).bin"

        echo -n "  fast_div / t=$threads: "

        timing_line=$(./histogram_fast_div "$bin_input" "$output_file" 2>&1 | grep "^TIMING,")
        mean=$(echo "$timing_line" | cut -d',' -f6)
        std=$(echo "$timing_line" | cut -d',' -f7)
        N_val=$(echo "$timing_line" | cut -d',' -f4)
        M_val=$(echo "$timing_line" | cut -d',' -f5)

        # Verify correctness: compare binary outputs directly, only invoke check.py on mismatch
        serial_ref="$RESULT_DIR/out_serial_$(basename $input .dat).bin"
        correct="yes"
        if cmp -s "$output_file" "$serial_ref"; then
            :  # byte-identical to verified serial output
        else
            # Convert to text and run check.py
            output_file_txt="${output_file%.bin}.dat"
            python3 convert_to_binary.py --reverse "$output_file" "$output_file_txt"
            if ! python3 check.py "$input" "$output_file_txt" 2>&1 | grep -q "CORRECT"; then
                correct="no"
            fi
            rm -f "$output_file_txt"
        fi

        echo "mean=${mean}s std=${std}s correct=${correct}"
        echo "fast_div,${threads},${N_val},${M_val},${mean},${std},${correct}" >> "$CSV"
    done
    echo ""
done

# ── Summary Table ────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo " Summary (speedup vs serial baseline)"
echo "═══════════════════════════════════════════════════"

python3 - "$CSV" <<'PYEOF'
import sys, csv

rows = list(csv.DictReader(open(sys.argv[1])))

# Find serial baselines by N
serial_times = {}
for r in rows:
    if r['strategy'] == 'serial' and r['threads'] == '1':
        serial_times[r['N']] = float(r['mean_sec'])

# Print table
print(f"\n{'Strategy':<12} {'Threads':>7} {'N':>12} {'Mean(s)':>10} {'Std(s)':>10} {'Speedup':>8} {'OK':>4}")
print("─" * 70)
for r in rows:
    base = serial_times.get(r['N'], float(r['mean_sec']))
    t = float(r['mean_sec'])
    speedup = base / t if t > 0 else 0
    ok = "✓" if r['correct'] == 'yes' else "✗"
    print(f"{r['strategy']:<12} {r['threads']:>7} {r['N']:>12} {r['mean_sec']:>10} {r['std_sec']:>10} {speedup:>7.2f}x {ok:>4}")
PYEOF

echo ""
echo "Results saved to: $CSV"
echo ""

# ── Cleanup artifacts (keep CSVs and inputs for re-runs) ─────────────────────
echo "[CLEANUP] Removing output artifacts and binaries..."
"$(dirname "$0")/cleanup.sh"
echo ""
