#!/bin/bash
# false_sharing_bench.sh — False Sharing Experiment Sweep
# Runs 4 experiments using a SINGLE input file with varying M values:
#   A: Dense M sweep (primary data)
#   B: Paired analysis (extracted from A, no extra runs)
#   C: Thread count scaling (M=32 vs M=33)
#   D: Padding threshold (M=33, threads=4)
#
# The same input file is reused across all experiments; M is overridden via CLI.
#
# Usage: ./false_sharing_bench.sh [output_dir]

set -euo pipefail

RESULT_DIR="${1:-results}"
mkdir -p "$RESULT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_A="$RESULT_DIR/fs_expA_dense_${TIMESTAMP}.csv"
CSV_C="$RESULT_DIR/fs_expC_threads_${TIMESTAMP}.csv"
CSV_D="$RESULT_DIR/fs_expD_padding_${TIMESTAMP}.csv"

# ── Configuration ─────────────────────────────────────────────────────────────
# Use N=100M, uniform [0,1)
FS_N=100000000
FS_SEED=42
FS_M_FILE=256          # M stored in the input file (arbitrary, will be overridden)

# Single input file for all experiments
INPUT_FILE="$RESULT_DIR/input_fs.dat"
INPUT_FILE_BIN="$RESULT_DIR/input_fs.bin"

# Experiment A: Dense M sweep
M_VALUES="8 9 12 16 17 18 20 24 32 33 36 48 64 65 72"
EXP_A_THREADS=16
EXP_A_PADDING=0

# Experiment C: Thread scaling
EXP_C_M_ALIGNED=32
EXP_C_M_MISALIGNED=33
EXP_C_THREADS="1 2 4 8 16"
EXP_C_PADDING=0

# Experiment D: Padding threshold
EXP_D_M=33
EXP_D_THREADS=16
EXP_D_PADDINGS="0 4 8 16 32 64 128"

NUM_RUNS=20
WARMUP=2
# ──────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════"
echo " False Sharing Experiment Suite"
echo " N=$FS_N, seed=$FS_SEED"
echo " Single input file, M overridden per run"
echo "═══════════════════════════════════════════════════"
echo ""

# Build
echo "[BUILD] Compiling false_sharing_exp..."
make false_sharing_exp 2>&1 | grep -v "^make\[" || true
echo ""

# ── Generate single input file ───────────────────────────────────────────────
echo "[GEN] Generating single input file (N=$FS_N, M_file=$FS_M_FILE)..."
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

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A: Dense M Sweep
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Experiment A: Dense M Sweep (threads=$EXP_A_THREADS, padding=$EXP_A_PADDING)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "M,padding_bytes,threads,N,mean_sec,std_sec,spill_bytes,aligned" > "$CSV_A"

for m in $M_VALUES; do
    export OMP_NUM_THREADS=$EXP_A_THREADS

    echo -n "  M=$m: "
    output=$(./false_sharing_exp "$INPUT_FILE_BIN" $EXP_A_PADDING $EXP_A_THREADS $NUM_RUNS $WARMUP $m 2>&1 > "$RESULT_DIR/out_fs_m${m}.dat")
    timing_line=$(echo "$output" | grep "^FALSESHARE,")

    mean=$(echo "$timing_line" | cut -d',' -f6)
    std=$(echo "$timing_line" | cut -d',' -f7)
    N_val=$(echo "$timing_line" | cut -d',' -f5)

    # Calculate spill and alignment
    spill=$(( (m * 4) % 64 ))
    aligned="no"
    if [ $spill -eq 0 ]; then aligned="yes"; fi

    echo "mean=${mean}s std=${std}s spill=${spill}B aligned=${aligned}"
    echo "${m},${EXP_A_PADDING},${EXP_A_THREADS},${N_val},${mean},${std},${spill},${aligned}" >> "$CSV_A"
done
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B: Paired Analysis (extracted from A)
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Experiment B: Paired Analysis (from Experiment A data)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 - "$CSV_A" <<'PYEOF'
import csv, sys

rows = {int(r['M']): r for r in csv.DictReader(open(sys.argv[1]))}

sub1 = [
    (8, 9, "Minimal spill (+1 int)"),
    (16, 17, "Minimal spill (+1 int)"),
    (32, 33, "Minimal spill (+1 int)"),
    (64, 65, "Minimal spill (+1 int)"),
]

sub2 = [
    (8, 9, "Spill = 4B"),
    (16, 18, "Spill = 8B"),
    (32, 36, "Spill = 16B"),
    (64, 72, "Spill = 32B"),
]

def print_sub(name, pairs):
    print(f"\n  {name}")
    print(f"  {'Pair':<6} {'Aligned':<10} {'Misaligned':<12} {'Time(A)':<12} {'Time(M)':<12} {'Penalty':<10} {'Note'}")
    print("  " + "─" * 83)
    for i, (aligned_m, misaligned_m, note) in enumerate(pairs, 1):
        if aligned_m in rows and misaligned_m in rows:
            ta = float(rows[aligned_m]['mean_sec'])
            tm = float(rows[misaligned_m]['mean_sec'])
            penalty = (tm - ta) / ta * 100 if ta > 0 else 0
            print(f"    {i:<4} M={aligned_m:<6} M={misaligned_m:<8} {ta:<12.6f} {tm:<12.6f} {penalty:>+7.1f}%   {note}")
        else:
            missing = []
            if aligned_m not in rows: missing.append(f"M={aligned_m}")
            if misaligned_m not in rows: missing.append(f"M={misaligned_m}")
            print(f"    {i:<4} M={aligned_m:<6} M={misaligned_m:<8} {'N/A':<12} {'N/A':<12} {'N/A':<10} {note} (missing: {', '.join(missing)})")

print_sub("Sub-setting 1: Minimal spill (aligned vs aligned+1)", sub1)
print_sub("Sub-setting 2: Increasing spill amount", sub2)
print()
PYEOF
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C: Thread Count Scaling
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Experiment C: Thread Scaling (M=$EXP_C_M_ALIGNED vs M=$EXP_C_M_MISALIGNED, padding=$EXP_C_PADDING)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "M,padding_bytes,threads,N,mean_sec,std_sec" > "$CSV_C"

for m in $EXP_C_M_ALIGNED $EXP_C_M_MISALIGNED; do
    echo "  M=$m:"
    for threads in $EXP_C_THREADS; do
        export OMP_NUM_THREADS=$threads
        echo -n "    t=$threads: "

        output=$(./false_sharing_exp "$INPUT_FILE_BIN" $EXP_C_PADDING $threads $NUM_RUNS $WARMUP $m 2>&1 > /dev/null)
        timing_line=$(echo "$output" | grep "^FALSESHARE,")
        mean=$(echo "$timing_line" | cut -d',' -f6)
        std=$(echo "$timing_line" | cut -d',' -f7)
        N_val=$(echo "$timing_line" | cut -d',' -f5)

        echo "mean=${mean}s std=${std}s"
        echo "${m},${EXP_C_PADDING},${threads},${N_val},${mean},${std}" >> "$CSV_C"
    done
done
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D: Padding Threshold
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Experiment D: Padding Threshold (M=$EXP_D_M, threads=$EXP_D_THREADS)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "M,padding_bytes,threads,N,mean_sec,std_sec" > "$CSV_D"

export OMP_NUM_THREADS=$EXP_D_THREADS

for pad in $EXP_D_PADDINGS; do
    echo -n "  padding=${pad}B: "

    output=$(./false_sharing_exp "$INPUT_FILE_BIN" $pad $EXP_D_THREADS $NUM_RUNS $WARMUP $EXP_D_M 2>&1 > /dev/null)
    timing_line=$(echo "$output" | grep "^FALSESHARE,")
    mean=$(echo "$timing_line" | cut -d',' -f6)
    std=$(echo "$timing_line" | cut -d',' -f7)
    N_val=$(echo "$timing_line" | cut -d',' -f5)

    echo "mean=${mean}s std=${std}s"
    echo "${EXP_D_M},${pad},${EXP_D_THREADS},${N_val},${mean},${std}" >> "$CSV_D"
done
echo ""

# ── Final Summary ────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo " False Sharing Experiments Complete"
echo "═══════════════════════════════════════════════════"
echo " Input file:             $INPUT_FILE"
echo " Exp A (dense sweep):    $CSV_A"
echo " Exp C (thread scaling): $CSV_C"
echo " Exp D (padding):        $CSV_D"
echo ""
echo " Key finding preview:"

python3 - "$CSV_A" <<'PYEOF'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
aligned = [r for r in rows if r['aligned'] == 'yes']
misaligned = [r for r in rows if r['aligned'] == 'no']

if aligned and misaligned:
    avg_aligned = sum(float(r['mean_sec']) for r in aligned) / len(aligned)
    avg_misaligned = sum(float(r['mean_sec']) for r in misaligned) / len(misaligned)
    penalty = (avg_misaligned - avg_aligned) / avg_aligned * 100
    print(f"   Avg aligned time:    {avg_aligned:.6f}s ({len(aligned)} configs)")
    print(f"   Avg misaligned time: {avg_misaligned:.6f}s ({len(misaligned)} configs)")
    print(f"   Average penalty:     {penalty:+.1f}%")
PYEOF

echo ""

# ── Cleanup artifacts (keep CSVs and inputs for re-runs) ─────────────────────
echo "[CLEANUP] Removing output artifacts and binaries..."
"$(dirname "$0")/cleanup.sh"
