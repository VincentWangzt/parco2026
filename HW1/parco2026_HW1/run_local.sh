#!/bin/bash
# run_local.sh — Full local experiment runner for PARCO HW1
# Mirrors the workflow of run.slurm but for local execution.
# Records full hardware/system details, then runs benchmark + false sharing experiments.
#
# Usage: ./run_local.sh [output_dir]

set -euo pipefail

RESULT_DIR="${1:-results}"
mkdir -p "$RESULT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM & HARDWARE INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  PARCO HW1 — Full Local Experiment                                  ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "═══════════════════════════════════════════════════"
echo " System Information"
echo "═══════════════════════════════════════════════════"
echo ""

echo "── General ──"
echo "  Hostname:       $(hostname)"
echo "  Date:           $(date)"
echo "  User:           $(whoami)"
echo "  OS:             $(uname -s -r -m)"
if [ -f /etc/os-release ]; then
    echo "  Distribution:   $(. /etc/os-release && echo "$PRETTY_NAME")"
fi
echo "  Kernel:         $(uname -r)"
echo ""

echo "── CPU ──"
if [ -f /proc/cpuinfo ]; then
    echo "  Model:          $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
    echo "  Cores (phys):   $(grep 'cpu cores' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "  Threads (log):  $(nproc)"
    echo "  Sockets:        $(grep 'physical id' /proc/cpuinfo | sort -u | wc -l)"
    echo "  MHz:            $(grep -m1 'cpu MHz' /proc/cpuinfo | cut -d: -f2 | xargs)"
    echo "  Flags (select): $(grep -m1 'flags' /proc/cpuinfo | grep -oE '(sse4_2|avx|avx2|avx512f|fma)' | tr '\n' ' ')"
else
    echo "  (no /proc/cpuinfo available)"
    echo "  CPUs: $(nproc)"
fi
echo ""

echo "── Cache ──"
if command -v lscpu &>/dev/null; then
    lscpu | grep -i 'cache' | sed 's/^/  /'
elif [ -d /sys/devices/system/cpu/cpu0/cache ]; then
    for idx in /sys/devices/system/cpu/cpu0/cache/index*; do
        level=$(cat "$idx/level")
        type=$(cat "$idx/type")
        size=$(cat "$idx/size")
        line=$(cat "$idx/coherency_line_size" 2>/dev/null || echo "?")
        echo "  L${level} (${type}): ${size}, line=${line}B"
    done
else
    echo "  (cache info not available)"
fi
echo ""

echo "── Memory ──"
if [ -f /proc/meminfo ]; then
    echo "  Total:          $(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
    echo "  Available:      $(grep MemAvailable /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
else
    echo "  (memory info not available)"
fi
echo ""

echo "── Compiler ──"
echo "  g++ version:    $(g++ --version | head -1)"
echo "  g++ path:       $(which g++)"
echo "  C++ standard:   C++11 (as per Makefile)"
echo "  Opt flags:      -O3 -march=native"
echo ""

echo "── OpenMP ──"
OMP_VER=$(g++ -fopenmp -dM -E - < /dev/null 2>/dev/null | grep _OPENMP | awk '{print $3}')
echo "  _OPENMP:        ${OMP_VER:-unknown}"
# Translate date code to version
case "$OMP_VER" in
    201511) echo "  Version:        4.5" ;;
    201811) echo "  Version:        5.0" ;;
    202011) echo "  Version:        5.1" ;;
    202111) echo "  Version:        5.2" ;;
    *) echo "  Version:        (date code $OMP_VER)" ;;
esac
echo "  OMP_PROC_BIND:  ${OMP_PROC_BIND:-not set}"
echo "  OMP_PLACES:     ${OMP_PLACES:-not set}"
echo ""

echo "── Python ──"
echo "  python3:        $(python3 --version 2>&1)"
echo "  numpy:          $(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'not installed')"
echo ""

# Save system info to file for records
SYSINFO_FILE="$RESULT_DIR/sysinfo_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "PARCO HW1 — System Info"
    echo "========================"
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -a)"
    [ -f /etc/os-release ] && . /etc/os-release && echo "Distribution: $PRETTY_NAME"
    echo ""
    echo "CPU:"
    if [ -f /proc/cpuinfo ]; then
        grep -m1 'model name' /proc/cpuinfo
        echo "Logical CPUs: $(nproc)"
    fi
    echo ""
    echo "Cache:"
    command -v lscpu &>/dev/null && lscpu | grep -i 'cache'
    echo ""
    echo "Memory:"
    [ -f /proc/meminfo ] && grep -E '^(MemTotal|MemAvailable)' /proc/meminfo
    echo ""
    echo "Compiler: $(g++ --version | head -1)"
    echo "OpenMP _OPENMP: $OMP_VER"
} > "$SYSINFO_FILE"
echo "[INFO] System info saved to: $SYSINFO_FILE"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Thread affinity for consistent results (mirrors run.slurm)
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Generate inputs if needed
[ -f input1.dat ] || python3 gen_input.py
[ -f input2.dat ] || python3 gen_input.py

# Build
echo "[BUILD] Compiling all targets..."
make all
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Running Strategy Benchmark                      ║"
echo "╚═══════════════════════════════════════════════════╝"
chmod +x benchmark.sh
./benchmark.sh "$RESULT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
#  FALSE SHARING EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Running False Sharing Experiment                ║"
echo "╚═══════════════════════════════════════════════════╝"
chmod +x false_sharing_bench.sh
./false_sharing_bench.sh "$RESULT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  All Experiments Complete                        ║"
echo "╚═══════════════════════════════════════════════════╝"
echo "  Finished at: $(date)"
echo "  Results in:  $RESULT_DIR/"
echo ""
ls -la "$RESULT_DIR"/*.csv 2>/dev/null || echo "  (no CSV files found)"
echo ""
