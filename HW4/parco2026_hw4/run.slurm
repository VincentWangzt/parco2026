#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p gpu
#SBATCH -J hw4_life
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 30:00

set -euo pipefail
cd "$(dirname "$0")"

RUNS=${RUNS:-11}

# Pretty banner, repeated for every variant inside every case.
banner() { echo; echo "==== $* ===="; }

# Run one variant on one (N, T) case and verify the output grid against
# the serial reference produced earlier in the same case block.
run_variant() {
    local label=$1 N=$2 T=$3
    shift 3
    banner "$label  (N=$N, T=$T)"
    rm -f "life_N${N}_T${T}.txt"
    "$@"
    if [[ ! -f "life_N${N}_T${T}.txt" ]]; then
        echo "ERROR: [$label] output file life_N${N}_T${T}.txt not found" >&2
        exit 1
    fi
    python3 verify.py "life_N${N}_T${T}.txt"
}

make clean
make

# ---- Case S : small grid, used for the screenshot ----------
N=16; T=4
run_variant "Case S serial"       $N $T ./serial $N $T
run_variant "Case S MPI -np 4"    $N $T mpirun -np 4 ./parallel_mpi $N $T
run_variant "Case S CUDA"         $N $T ./parallel_cuda $N $T

# ---- Case M : non-divisible grid, exercises the row-remainder logic --------
N=257; T=100
run_variant "Case M serial"       $N $T ./serial $N $T
run_variant "Case M MPI -np 4"    $N $T mpirun -np 4 ./parallel_mpi $N $T
run_variant "Case M CUDA"         $N $T ./parallel_cuda $N $T

# ---- Case L : full benchmark sweep -----------------------------------------
# Each variant is timed RUNS times; raw per-run lines are printed verbatim,
# and a single median + speedup-vs-serial is reported at the end.

# pull the numeric ms value out of a "<...> compute time: <X> ms" line
extract_ms() {
    awk '{ for (k = 1; k <= NF; k++) if ($k == "ms") { print $(k - 1); exit } }'
}

# median of newline-separated numbers on stdin
median() {
    sort -g | awk '
        { a[NR] = $1 }
        END {
            if (NR == 0) { exit 1 }
            if (NR % 2)  { print a[(NR + 1) / 2] }
            else         { printf "%.6f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2 }
        }'
}

# args: label cmd...
bench_variant() {
    local label=$1
    shift
    local cmd=("$@")

    local tmp
    tmp=$(mktemp)
    local i line t
    for ((i = 1; i <= RUNS; i++)); do
        line=$("${cmd[@]}" 2>/dev/null | grep "compute time:" || true)
        if [[ -z $line ]]; then
            echo "[$label] no compute-time line on run $i" >&2
            rm -f "$tmp"
            exit 1
        fi
        t=$(extract_ms <<<"$line")
        printf 'Case L %-9s run %2d: %s ms\n' "$label" "$i" "$t"
        printf '%s\n' "$t" >>"$tmp"
    done

    local med
    med=$(median <"$tmp")
    rm -f "$tmp"
    eval "MED_${label}=\$med"
}

banner "Case L benchmark (raw timings, $RUNS runs per variant)"
N=1024; T=200
bench_variant serial   ./serial "$N" "$T"
bench_variant mpi_np1  mpirun -np 1 ./parallel_mpi "$N" "$T"
bench_variant mpi_np4  mpirun -np 4 ./parallel_mpi "$N" "$T"
bench_variant mpi_np16 mpirun -np 16 ./parallel_mpi "$N" "$T"
bench_variant cuda     ./parallel_cuda "$N" "$T"

# Per-case summary: one median + speedup line per variant.
banner "Case L summary"
base=$MED_serial
for label in serial mpi_np1 mpi_np4 mpi_np16 cuda; do
    med=$(eval echo "\$MED_$label")
    sp=$(awk -v b="$base" -v m="$med" 'BEGIN { printf "%.2f", b / m }')
    printf 'Case L %s: median=%s ms, speedup=%sx\n' "$label" "$med" "$sp"
done
