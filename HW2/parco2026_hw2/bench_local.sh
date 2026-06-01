#!/bin/bash
# bench_local.sh — local benchmark harness producing the same [bench]/time= log
# format that parse_results.py expects. Sourced by hand during the optimization
# iterations so we can quickly check effect of each change without going to
# SLURM.
#
# Usage:
#   ./bench_local.sh                              # default: NS="8192 16384", PS="1 2 4 8 16", REPS=5
#   NS=8192 PS=16 REPS=5 ./bench_local.sh         # a quick single-point check
#   EXES="parallel"  NS=8192 PS=16 REPS=5 ./bench_local.sh
#   OUT=bench_local_v1.out ./bench_local.sh
#
# What we do NOT do here:
#   - Phase 0 correctness (run separately with verify.py)
#   - mpiicc build (cluster only)
#
# Everything else mirrors bench.slurm so parse_results.py can ingest the
# resulting log file directly.

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load mpi/openmpi-x86_64 2>/dev/null || true

NS=${NS:-"8192 16384"}
PS=${PS:-"1 2 4 8 16"}
REPS=${REPS:-5}
EXES=${EXES:-"parallel"}              # which binaries to drive
OUT=${OUT:-"bench_local.out"}
# Local box: --oversubscribe avoids "not enough slots" errors;
# --mca pml ob1 --mca btl self,vader pins the transport to shared-memory only
# (the host's PSM3/UCX defaults fail in this container).
# NOTE: explored --bind-to core --map-by socket in v2 but it consistently
# degraded p={4,8} (e.g. p=8 16384: 35.8ms -> 59.3ms). The container's cpuset
# already restricts cores; OMPI's hard binding fights cgroup placement. So we
# do NOT bind on the local box. The cluster's bench.slurm uses SLURM's --cpu-bind.
MPI_OPTS=${MPI_OPTS:-"--oversubscribe --mca pml ob1 --mca btl self,vader"}
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

: > "$OUT"
{
  echo "=== local bench  $(date)"
  echo "=== EXES=$EXES NS=$NS PS=$PS REPS=$REPS MPI_OPTS=$MPI_OPTS"
  echo

  # serial_timed baseline (only if requested)
  if [[ " $EXES " == *" serial_timed "* ]]; then
    for N in $NS; do
      for r in $(seq 1 "$REPS"); do
        echo "[bench] serial,$N,1,$r"
        mpirun $MPI_OPTS -np 1 ./serial_timed "$N"
      done
    done
  fi

  # parallel sweeps for each binary in EXES
  for exe in $EXES; do
    [[ "$exe" == "serial_timed" ]] && continue
    [[ ! -x "./$exe" ]] && { echo "missing ./$exe — skipping" >&2; continue; }
    for N in $NS; do
      for p in $PS; do
        for r in $(seq 1 "$REPS"); do
          echo "[bench] $exe,$N,$p,$r"
          mpirun $MPI_OPTS -np "$p" "./$exe" "$N"
        done
      done
    done
  done

  echo "=== done $(date) ==="
} | tee -a "$OUT"
