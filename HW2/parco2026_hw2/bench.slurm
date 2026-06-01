#!/bin/bash
#SBATCH -p compute
#SBATCH -J mpi-gemv-bench
#SBATCH --nodes=1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks-per-node 16
#SBATCH --time 00:60:00
#SBATCH --output bench.%j.out
#SBATCH --mem 21844M
#SBATCH --export ALL
#SBATCH --get-user-env
#
# bench.slurm — single end-to-end job for the assignment.
#
#   Phase 0  correctness for cases S (N=32) and M (N=1000)
#   Phase 1  case L performance: serial vs 16-process parallel (medians)
#   Phase 2  strong-scaling sweep p ∈ {1,2,4,8,16}
#
# Submit with:  sbatch bench.slurm
# Result:       bench.<JOBID>.out  +  y_N{32,1000,8192,16384}.txt
# Then locally: scp cluster:bench.<JOBID>.out .
#               python3 parse_results.py bench.<JOBID>.out
# parse_results.py builds bench_results.csv, prints the report tables,
# and (optionally) regenerates figs/scaling_*.png.
#
# Output format the parser depends on:
#   Each measured run is preceded by a single tag line of the form
#     [bench] <exe>,<N>,<p>,<run>
#   Then the program prints its `time=<seconds> s` line. parse_results.py
#   pairs every [bench] tag with the FIRST `time=` line that follows it.
#   Unmatched lines (Phase 0 / banners / verify.py) are ignored.
#
# Knobs: edit NS / PS / REPS below or override on submit, e.g.
#   sbatch --export=ALL,NS="8192",PS="16",REPS=5 bench.slurm

set -euo pipefail

NS=${NS:-"8192 16384"}      # matrix sizes for Phase 1 / 2
PS=${PS:-"1 2 4 8 16"}      # MPI process counts for strong scaling
REPS=${REPS:-5}             # repetitions per (exe, N, p)

echo "=== node:    $(hostname)"
echo "=== date:    $(date)"
echo "=== nproc:   $(nproc)"
echo "=== NS=$NS  PS=$PS  REPS=$REPS"
echo

# Clean rebuild with cluster's mpiicc (-O3 from Makefile).
make clean
make
echo

# -----------------------------------------------------------------------
# Phase 0 — correctness for cases S and M.
#   S (N=32):   diff serial vs parallel output bit-for-bit + verify.py
#   M (N=1000): not divisible by 16, exercises MPI_Gatherv path
# These runs are NOT timed measurements — Phase 0 lines have no [bench]
# tag, so parse_results.py skips them.
# -----------------------------------------------------------------------
echo "============================================================"
echo "PHASE 0 — correctness (cases S and M)"
echo "============================================================"

echo "--- CASE S (N = 32) ---"
./serial 32
cp y_N32.txt y_N32_serial.txt
mpirun -np 16 ./parallel 32
diff y_N32.txt y_N32_serial.txt && echo "  >>> parallel == serial bit-for-bit"
python3 verify.py y_N32.txt
echo

echo "--- CASE M (N = 1000) ---"
mpirun -np 16 ./parallel 1000
python3 verify.py y_N1000.txt
echo

# -----------------------------------------------------------------------
# Phase 1 — case L performance: serial vs 16-process parallel.
# Five reps each. Median speedup goes into the report's headline number.
# -----------------------------------------------------------------------
echo "============================================================"
echo "PHASE 1 — case L (serial vs parallel p=16)"
echo "============================================================"
for N in $NS; do
  for r in $(seq 1 "$REPS"); do
    echo "[bench] serial,$N,1,$r"
    mpirun -np 1 ./serial_timed "$N"
  done
  for r in $(seq 1 "$REPS"); do
    echo "[bench] parallel,$N,16,$r"
    mpirun -np 16 ./parallel "$N"
  done
done
echo

# -----------------------------------------------------------------------
# Phase 2 — strong scaling (parallel only).
# Sweep p ∈ PS. The p=1 baseline IS the parallel build with 1 rank, NOT
# the serial build, so S(p)=T(1)/T(p) measures only the parallelism (no
# compiler/init bias).
# -----------------------------------------------------------------------
echo "============================================================"
echo "PHASE 2 — strong-scaling sweep"
echo "============================================================"
for N in $NS; do
  for p in $PS; do
    for r in $(seq 1 "$REPS"); do
      echo "[bench] parallel,$N,$p,$r"
      mpirun -np "$p" ./parallel "$N"
    done
  done
done
echo

# -----------------------------------------------------------------------
# Verify the largest result file so the data CSV is trustworthy.
# -----------------------------------------------------------------------
for N in $NS; do
  python3 verify.py "y_N${N}.txt"
done

echo "=== done at $(date) ==="
