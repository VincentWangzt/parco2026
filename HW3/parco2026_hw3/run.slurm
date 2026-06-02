#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p gpu
#SBATCH -J hw3_scan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 10:00


# 算例 S：小规模，用于正确性截图
./serial 32

# 算例 M：非 2 的幂规模，用于自动验证
./serial 1000003
python3 verify.py scan_N1000003.txt

# 算例 L：性能测试，记录串行时间、CUDA 时间和加速比
./serial 8388608
