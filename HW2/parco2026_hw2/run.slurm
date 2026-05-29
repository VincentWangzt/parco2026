#!/bin/bash
#CBATCH -p compute
#CBATCH --qos low
#CBATCH -J mpitest
#CBATCH -N 1
#CBATCH --ntasks-per-node 4
#CBATCH -c 1
#CBATCH --time 00:30:00
#CBATCH --output job.%j.out
#CBATCH --mem 21844M
#CBATCH --export ALL
#CBATCH --get-user-env

# 算例 S：小规模，用于正确性截图（生成 y_N32.txt）
mpirun -np 16 ./serial 32

# 算例 M：N=1000 不被 16 整除，必须使用 MPI_Gatherv（生成 y_N1000.txt）
mpirun -np 16 ./parallel 1000

# 算例 L：性能测试，记录运行时间用于加速比分析（生成 y_N8192.txt）
mpirun -np 16 ./parallel 8192
