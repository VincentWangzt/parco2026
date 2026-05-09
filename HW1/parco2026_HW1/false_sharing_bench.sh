#!/bin/bash
# false_sharing_bench.sh — False Sharing Padding Sweep
#
# For each M value, sweep padding_bytes and thread counts to demonstrate
# the false sharing effect and its elimination via padding.
#
# M values chosen for their cache-line alignment properties:
#   M=8   → stride=32B (half a cache line)  → maximum false sharing
#   M=33  → stride=132B (spill=4B)          → moderate false sharing
#   M=128 → stride=512B (exactly 8 lines)   → no false sharing (control)
#
# Usage: ./false_sharing_bench.sh [output_dir]

set -euo pipefail

RESULT_DIR="${1:-results}"
mkdir -p "$RESULT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_OUT="$RESULT_DIR/fs_padding_sweep_${TIMESTAMP}.csv"

# ── Configuration ─────────────────────────────────────────────────────────────
FS_N=100000000
FS_SEED=42
FS_M_FILE=256          # M stored in the input file (will be overridden per run)

# Single input file for all experiments
INPUT_FILE="$RESULT_DIR/input_fs.dat"
INPUT_FILE_BIN="$RESULT_DIR/input_fs.bin"

# Experiment parameters
M_VALUES="8 33 128"
PADDING_VALUES="0 4 8 16 32 64 128 256 512"
THREAD_VALUES="4 16"

NUM_RUNS=20
WARMUP=2
# ──────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════"
echo " False Sharing Padding Sweep"
echo " N=$FS_N, seed=$FS_SEED"
echo " M values: $M_VALUES"
echo " Padding:  $PADDING_VALUES"
echo " Threads:  $THREAD_VALUES"
echo " Runs=$NUM_RUNS, Warmup=$WARMUP"
echo "═══════════════════════════════════════════════════"
echo ""

# Build
echo "[BUILD] Compiling false_sharing_exp..."
make false_sharing_exp 2>&1 | grep -v "^make\[" || true
echo ""

# ── Generate single input file ───────────────────────────────────────────────
echo "[GEN] Generating input file (N=$FS_N, M_file=$FS_M_FILE)..."
if [ ! -f "$INPUT_FILE" ]; then
    python3 gen_input.py $FS_N $FS_M_FILE 0.0 1.0 "$INPUT_FILE" $FS_SEED
fi
echo "  -> $INPUT_FILE"

# Convert to binary for faster I/O
if [ ! -f "$INPUT_FILE_BIN" ]; then
    echo "[CONVERT] $INPUT_FILE -> $INPUT_FILE_BIN"
    python3 convert_to_binary.py "$INPUT_FILE" "$INPUT_FILE_BIN"
fi
echo ""

# ── Run Padding Sweep ────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Padding Sweep: M × padding_bytes × threads"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "M,padding_bytes,threads,N,mean_sec,std_sec" > "$CSV_OUT"

for m in $M_VALUES; do
    stride_bytes=$(( m * 4 ))
    spill=$(( stride_bytes % 64 ))
    echo ""
    echo "  ┌─ M=$m (stride=${stride_bytes}B, spill=${spill}B)"

    for threads in $THREAD_VALUES; do
        echo "  │  threads=$threads"
        export OMP_NUM_THREADS=$threads

        for pad in $PADDING_VALUES; do
            echo -n "  │    pad=${pad}B: "

            output=$(./false_sharing_exp "$INPUT_FILE_BIN" $pad $threads $NUM_RUNS $WARMUP $m 2>&1 > /dev/null)
            timing_line=$(echo "$output" | grep "^FALSESHARE,")
            mean=$(echo "$timing_line" | cut -d',' -f6)
            std=$(echo "$timing_line" | cut -d',' -f7)
            N_val=$(echo "$timing_line" | cut -d',' -f5)

            echo "mean=${mean}s std=${std}s"
            echo "${m},${pad},${threads},${N_val},${mean},${std}" >> "$CSV_OUT"
        done
    done
    echo "  └─"
done
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo " False Sharing Padding Sweep Complete"
echo "═══════════════════════════════════════════════════"
echo " Output: $CSV_OUT"
echo ""

python3 - "$CSV_OUT" <<'PYEOF'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))

print(" Summary by M value (threads=16):")
print(" ─────────────────────────────────────────────────")

for m_val in ['8', '33', '128']:
    subset = [r for r in rows if r['M'] == m_val and r['threads'] == '16']
    if not subset:
        continue
    t_nopad = float(subset[0]['mean_sec'])  # padding=0
    t_maxpad = float(subset[-1]['mean_sec'])  # max padding
    t_min = min(float(r['mean_sec']) for r in subset)
    speedup = t_nopad / t_min if t_min > 0 else 0
    print(f"   M={m_val:>3}: no-pad={t_nopad:.4f}s  best={t_min:.4f}s  speedup={speedup:.1f}x")

print()
PYEOF

echo ""

# ── Cleanup artifacts ────────────────────────────────────────────────────────
echo "[CLEANUP] Removing output artifacts and binaries..."
"$(dirname "$0")/cleanup.sh"
