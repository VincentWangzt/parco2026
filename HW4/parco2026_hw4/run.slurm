#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p gpu
#SBATCH -J hw4_life
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 30:00

export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader

# 算例 S：小规模，用于正确性截图和可视化
echo "==== Case S serial ===="
./serial 16 4
python3 verify.py life_N16_T4.txt
python3 visual.py life_N16_T4.txt life_N16_T4.png

echo "==== Case S MPI -np 4 ===="
mpirun --allow-run-as-root -np 4 ./parallel_mpi 16 4
python3 verify.py life_N16_T4.txt

echo "==== Case S CUDA ===="
./parallel_cuda 16 4
python3 verify.py life_N16_T4.txt

echo "==== Case S MPI+CUDA -np 4 ===="
mpirun --allow-run-as-root -x LD_LIBRARY_PATH -np 4 ./parallel_mpi_cuda 16 4
python3 verify.py life_N16_T4.txt

# 算例 M：非整除规模，用于自动验证
echo "==== Case M serial ===="
./serial 257 100
python3 verify.py life_N257_T100.txt

echo "==== Case M MPI -np 4 ===="
mpirun --allow-run-as-root -np 4 ./parallel_mpi 257 100
python3 verify.py life_N257_T100.txt

echo "==== Case M CUDA ===="
./parallel_cuda 257 100
python3 verify.py life_N257_T100.txt

echo "==== Case M MPI+CUDA -np 2 ===="
mpirun --allow-run-as-root -x LD_LIBRARY_PATH -np 2 ./parallel_mpi_cuda 257 100
python3 verify.py life_N257_T100.txt

# 算例 L：性能测试，记录串行时间、并行时间和加速比
echo "==== Case L benchmarks (median over 11 runs) ===="
python3 bench_all.py
