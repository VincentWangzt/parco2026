#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p gpu
#SBATCH -J hw4_life
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 20:00



# 算例 S：小规模，用于正确性截图和可视化
./serial 16 4
python3 verify.py life_N16_T4.txt

# 如果选择 MPI 实现，可以取消下面一行注释
# mpirun -np 4 ./parallel_mpi 16 4

# 算例 M：非整除规模，用于自动验证
./serial 257 100

python3 verify.py life_N257_T100.txt

# 算例 L：性能测试，记录串行时间、并行时间和加速比
./serial 1024 200
