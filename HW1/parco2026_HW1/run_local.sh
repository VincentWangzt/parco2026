#!/bin/bash
# run_local.sh — Quick local execution helper
# Usage: ./run_local.sh [threads] [input] [strategy]

THREADS=${1:-4}
INPUT=${2:-input1.dat}
STRATEGY=${3:-padded}

# Generate input if needed
if [ ! -f "$INPUT" ]; then
    echo "[GEN] Generating input files..."
    python3 gen_input.py
fi

# Build
echo "[BUILD] Compiling..."
make all 2>&1 | grep -v "^make\[" || true
echo ""

echo "═══════════════════════════════════════════════════"
echo " strategy=$STRATEGY  threads=$THREADS  input=$INPUT"
echo "═══════════════════════════════════════════════════"
echo ""

# Run serial baseline
echo "── Serial Baseline ──"
./serial "$INPUT" output_local_serial.dat
echo ""

# Run selected strategy
echo "── Strategy: $STRATEGY (threads=$THREADS) ──"
export OMP_NUM_THREADS=$THREADS
./histogram_omp "$STRATEGY" "$INPUT" output_local_strategy.dat
python3 check.py "$INPUT" output_local_strategy.dat
echo ""

# Run fast version
echo "── Fast Version (threads=$THREADS) ──"
./histogram_fast "$INPUT" output_local_fast.dat
python3 check.py "$INPUT" output_local_fast.dat
echo ""

# Cleanup
rm -f output_local_serial.dat output_local_strategy.dat output_local_fast.dat
