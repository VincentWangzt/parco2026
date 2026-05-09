#!/bin/bash
#SBATCH -o job_%j.out
#SBATCH -p compute
#SBATCH -J parco_hw1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00

echo "═══════════════════════════════════════════════════"
echo " PARCO HW1 — Job $CRANE_JOB_ID on $(hostname)"
echo " Date: $(date)"
echo " CPUs: $SLURM_CPUS_PER_TASK"
echo " Mem:  $CRANE_MEM_PER_NODE"
echo "═══════════════════════════════════════════════════"
echo ""

# Thread affinity for consistent results
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Generate inputs if needed
[ -f input1.dat ] || python3 gen_input.py
[ -f input2.dat ] || python3 gen_input.py

# Build
make all

# Run full strategy benchmark
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Running Strategy Benchmark                      ║"
echo "╚═══════════════════════════════════════════════════╝"
chmod +x benchmark.sh
./benchmark.sh results_slurm

# Run false sharing experiment
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Running False Sharing Experiment                ║"
echo "╚═══════════════════════════════════════════════════╝"
chmod +x false_sharing_bench.sh
./false_sharing_bench.sh results_slurm

echo ""
echo "Done at $(date)"
