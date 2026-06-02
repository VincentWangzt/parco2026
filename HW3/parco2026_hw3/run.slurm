#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -e job_%j_%N.err
#SBATCH -p gpu
#SBATCH -J hw3_scan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 30:00

# -----------------------------------------------------------------------------
# Cluster benchmark script for HW3 CUDA inclusive prefix sum.
#
# Usage:
#     sbatch run.slurm
#
# This script
#   1. Loads CUDA / GCC modules (best-effort; harmless if absent).
#   2. (Re)builds `serial` and `parallel`.
#   3. Runs the three required cases (S, M, L).
#   4. Verifies S and M with verify.py.
#   5. For case L, repeats BOTH serial and parallel 11 times (parallel has
#      one extra warm-up run). Takes the MEDIAN of the 11 timings, which
#      is robust to one or two stray outliers and does not require throwing
#      away samples by hand. Reports the speedup vs. that median.
# -----------------------------------------------------------------------------

set -e
set -u

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")"

# --- Try to load common module names; ignore if not present ----------------
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
  module load gcc       2>/dev/null || true
fi

echo "============================================================"
echo "Host    : $(hostname)"
echo "Date    : $(date)"
echo "CWD     : $(pwd)"
echo "nvcc    : $(command -v nvcc || echo 'NOT FOUND')"
nvcc --version 2>/dev/null | tail -1 || true
echo "------------------------------------------------------------"
nvidia-smi 2>/dev/null | head -20 || true
echo "============================================================"

# --- Build ------------------------------------------------------------------
make clean
make

# --- Helpers ----------------------------------------------------------------
extract_serial_ms()      { grep -E "Serial compute time"        "$1" | awk '{print $4}'; }
extract_cuda_scan_ms()   { grep -E "CUDA scan \(kernels only\)" "$1" | awk '{print $5}'; }
extract_cuda_total_ms()  { grep -E "CUDA total"                 "$1" | awk '{print $4}'; }

# Median of an arbitrary-length list of numbers (one per line) on stdin.
# For an even count, returns the lower of the two middle values; for 11
# samples (the default below) we always have an exact middle.
median_stdin() { sort -g | awk '{a[NR]=$0} END{ if (NR==0) {print "NaN"} else if (NR%2) {print a[(NR+1)/2]} else {print a[NR/2]} }'; }

run_case() {
  local N="$1"
  local label="$2"
  echo
  echo "=================  Case ${label}  (N = ${N})  ================="

  # Serial
  local SERIAL_LOG="serial_N${N}.log"
  ./serial "$N" | tee "$SERIAL_LOG"

  # Parallel
  local CUDA_LOG="parallel_N${N}.log"
  ./parallel "$N" | tee "$CUDA_LOG"

  # Verify against the parallel output (parallel writes the file last).
  if [ -f "scan_N${N}.txt" ]; then
    python3 verify.py "scan_N${N}.txt"
  fi
}

# --- Case S: correctness, screenshot ---------------------------------------
run_case 32 "S"

# --- Case M: non-power-of-two, automatic verify ----------------------------
run_case 1000003 "M"

# --- Case L: performance & speedup -----------------------------------------
N_L=8388608
N_REPEAT=11
echo
echo "================  Case L  (N = ${N_L})  ================"

SERIAL_TIMES_FILE=$(mktemp)
CUDA_SCAN_TIMES_FILE=$(mktemp)
CUDA_TOTAL_TIMES_FILE=$(mktemp)

# ---- Serial: ${N_REPEAT} runs ----
echo "[Serial] running ${N_REPEAT} timed iterations..."
for i in $(seq 1 ${N_REPEAT}); do
  ./serial "$N_L" > "serial_L_run${i}.log"
  T=$(extract_serial_ms "serial_L_run${i}.log")
  printf "Serial run %2d: %s ms\n" "$i" "$T"
  echo "$T" >> "$SERIAL_TIMES_FILE"
done

# Verify with the serial output before parallel overwrites the file.
python3 verify.py "scan_N${N_L}.txt"

# ---- Parallel: 1 warmup + ${N_REPEAT} timed runs ----
echo "[Parallel] warmup + ${N_REPEAT} timed iterations..."
./parallel "$N_L" > /dev/null  # warmup
for i in $(seq 1 ${N_REPEAT}); do
  LOG="parallel_L_run${i}.log"
  ./parallel "$N_L" > "$LOG"
  T_SCAN=$(extract_cuda_scan_ms "$LOG")
  T_TOTAL=$(extract_cuda_total_ms "$LOG")
  printf "CUDA run %2d: scan=%s ms, total=%s ms\n" "$i" "$T_SCAN" "$T_TOTAL"
  echo "$T_SCAN"  >> "$CUDA_SCAN_TIMES_FILE"
  echo "$T_TOTAL" >> "$CUDA_TOTAL_TIMES_FILE"
done

# Verify L again with the parallel output.
python3 verify.py "scan_N${N_L}.txt"

SERIAL_MEDIAN=$(median_stdin     < "$SERIAL_TIMES_FILE")
CUDA_MEDIAN_SCAN=$(median_stdin  < "$CUDA_SCAN_TIMES_FILE")
CUDA_MEDIAN_TOTAL=$(median_stdin < "$CUDA_TOTAL_TIMES_FILE")
SERIAL_MIN=$(sort -g "$SERIAL_TIMES_FILE"        | awk 'NR==1')
SERIAL_MAX=$(sort -g "$SERIAL_TIMES_FILE"        | awk 'END{print}')
CUDA_SCAN_MIN=$(sort -g "$CUDA_SCAN_TIMES_FILE"  | awk 'NR==1')
CUDA_SCAN_MAX=$(sort -g "$CUDA_SCAN_TIMES_FILE"  | awk 'END{print}')

echo
echo "================  Case L summary (N = ${N_L}, 11 runs)  ================"
printf "Serial      : median = %s ms   [min %s, max %s]\n" \
  "$SERIAL_MEDIAN" "$SERIAL_MIN" "$SERIAL_MAX"
printf "CUDA scan   : median = %s ms   [min %s, max %s]\n" \
  "$CUDA_MEDIAN_SCAN" "$CUDA_SCAN_MIN" "$CUDA_SCAN_MAX"
printf "CUDA total  : median = %s ms\n" "$CUDA_MEDIAN_TOTAL"
awk -v s="$SERIAL_MEDIAN" -v c="$CUDA_MEDIAN_SCAN" \
    'BEGIN { printf "Speedup (scan only) : %.2fx\n", s/c }'
awk -v s="$SERIAL_MEDIAN" -v c="$CUDA_MEDIAN_TOTAL" \
    'BEGIN { printf "Speedup (incl. D2H) : %.2fx\n", s/c }'

rm -f "$SERIAL_TIMES_FILE" "$CUDA_SCAN_TIMES_FILE" "$CUDA_TOTAL_TIMES_FILE"
echo "Done."
