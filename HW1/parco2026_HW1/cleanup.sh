#!/bin/bash
# cleanup.sh — Remove generated artifacts and input data
# Keeps CSV result files by default (the actual benchmark data you care about).
#
# Usage:
#   ./cleanup.sh              Remove binaries, outputs, and generated inputs
#   ./cleanup.sh --keep-inputs  Remove binaries and outputs only (keep .dat inputs)
#   ./cleanup.sh --all        Remove everything including CSV results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

KEEP_INPUTS=false
REMOVE_CSV=false

for arg in "$@"; do
    case "$arg" in
        --keep-inputs) KEEP_INPUTS=true ;;
        --all)         REMOVE_CSV=true ;;
        -h|--help)
            echo "Usage: $0 [--keep-inputs] [--all]"
            echo ""
            echo "  (default)       Remove binaries, output .dat files, and generated inputs"
            echo "  --keep-inputs   Keep input .dat files (only remove binaries + outputs)"
            echo "  --all           Also remove CSV result files"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "[CLEANUP] Removing build artifacts..."

# ── Binaries ─────────────────────────────────────────────────────────────────
rm -f serial histogram_omp histogram_fast false_sharing_exp
rm -f *.o

# ── Output .dat files (histogram results, not inputs) ────────────────────────
rm -f output*.dat
rm -f output*.bin
rm -f results/out_*.dat
rm -f results/out_*.bin


# ── Generated input data ─────────────────────────────────────────────────────
if [ "$KEEP_INPUTS" = false ]; then
    echo "[CLEANUP] Removing generated input files..."
    rm -f input1.dat input2.dat
    rm -f input1.bin input2.bin
    rm -f results/input_fs.dat
    rm -f results/input_fs.bin
fi

# ── CSV result files (only with --all) ───────────────────────────────────────
if [ "$REMOVE_CSV" = true ]; then
    echo "[CLEANUP] Removing CSV result files..."
    rm -f results/benchmark_*.csv
    rm -f results/fs_exp*.csv
fi

# ── Remove results/ dir if empty ─────────────────────────────────────────────
if [ -d results ] && [ -z "$(ls -A results 2>/dev/null)" ]; then
    rmdir results
fi

echo "[CLEANUP] Done."
