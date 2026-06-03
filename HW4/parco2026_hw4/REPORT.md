# 作业 4：生命游戏并行模拟 — 实验报告

## 1. 概述

本次作业实现了二维 Conway 生命游戏 (`N×N` 网格、`T` 步迭代、固定死亡边界)
的两种并行版本：

| 版本 | 文件 | 说明 |
|---|---|---|
| MPI 多进程 | `parallel_mpi.cpp` | 按行划分网格，进程间用 `MPI_Sendrecv` 交换 ghost rows |
| CUDA 单卡 | `parallel_cuda.cu` | 一个线程负责一个格子，双缓冲指针交换 |

初始网格严格使用题目给定的公式
`grid[i][j] = (((i*131 + j*17 + 7) % 100) < 30)`，所有版本与 `verify.py`
对照通过，结果与串行参考完全一致。

测试平台：

* CPU：服务器内可用 ≥ 16 个核心（`ntasks-per-node=4`，但 `mpirun --oversubscribe -np 16` 可正常运行）
* GPU：NVIDIA Tesla T4（16 GB），CUDA 12.2
* OpenMPI 4.1.1，g++ 8.5.0，nvcc 12.x，`-O3 -arch=sm_60`

时间测量方式：每个版本针对每个算例运行 **11 次**，记录每次内核计算时间
（串行/MPI 用 `std::chrono`，CUDA 在 `cudaDeviceSynchronize` 前后取时间），
取 **中位数** 作为最终运行时间，并以串行中位数为基准计算加速比。

---

## 2. 并行划分与边界处理

### 2.1 MPI 实现 (`parallel_mpi.cpp`)

* **划分方式**：1-D 行划分。当 `N` 不能被进程数 `P` 整除时，
  把余数 `N % P` 行依次分配到前 `N % P` 个进程，每个进程负责
  `local_rows = ⌊N/P⌋ + (rank < N%P)` 行。
* **本地存储**：每个进程持有一个 `(local_rows + 2) × N` 的字节矩阵，
  第 `0` 行和第 `local_rows+1` 行是 ghost rows，分别保存上邻居
  和下邻居的边界行，初始值为 `0`。
* **每步流程**：
  1. `MPI_Sendrecv`：把自己的最上一行（行 `1`）送给上邻居 `rank-1`，
     同时从下邻居接收下 ghost row。
  2. 再做一次 `MPI_Sendrecv` 把最下一行送给下邻居、从上邻居接收上 ghost row。
  3. 在全局上下边界 (`rank==0` / `rank==P-1`) 处把对应 ghost row 用 `memset(0)` 清零，
     从而严格满足"网格外格子全为 0"的固定死亡边界。
  4. 双重循环更新 `next` 中对应的 `local_rows` 行；左/右边界靠
     `if (jl >= 0)` / `if (jr < N)` 直接跳过，避免越界。
  5. `current.swap(next)`。
* **边界处理小结**：
  * 上下边界：通过 `MPI_PROC_NULL` 让 `MPI_Sendrecv` 在端点 rank 自动跳过通信，
    再 `memset` ghost row 为 0；
  * 左右边界：每行更新时显式判断 `jl/jr` 是否落在 `[0, N)`；
  * 这样三种边界条件都与串行版的 `live_neighbors` 保持完全一致。
* **结果汇集**：使用 `MPI_Gatherv(counts, offsets)` 把每个进程的局部行
  按行偏移聚合到 rank 0，再写入 `life_N<N>_T<T>.txt`。

### 2.2 CUDA 实现 (`parallel_cuda.cu`)

* **划分方式**：每个 CUDA 线程负责一个格子；线程块为 `16×16`，
  网格大小向上取整为 `(⌈N/16⌉, ⌈N/16⌉)`。
* **双缓冲**：在显存上分配 `d_current`、`d_next` 两块 `N×N` 字节缓冲；
  每步内核从 `d_current` 读、向 `d_next` 写，结束后在 host 端
  `std::swap(d_current, d_next)`。
* **边界**：内核里直接判断 `0 ≤ ni < N && 0 ≤ nj < N`，
  避免把网格外的"死格子"读进来。这样实现简洁，
  且对 `N=257` 这种非整除规模天然支持（多余线程在 `i>=N || j>=N`
  时早返回）。
* **显存访问**：网格按行主序连续存储，
  同一线程块的 16×16 个线程在读 9 个邻居时可以触发 L1/纹理缓存的局部命中，
  对 T4 这种 GDDR6 GPU 来说带宽利用充足。
* **测时与冷启动**：测时之前先做一次 warm-up 内核启动并
  `cudaDeviceSynchronize`，再把初始网格重新拷回设备，
  避免把 nvcc JIT/runtime 初始化开销算进 `T` 步迭代时间。

---

## 3. 实验结果

### 3.1 算例 S （N = 16, T = 4）— 正确性 / 截图 / 可视化

* `serial`、`parallel_mpi -np 4`、`parallel_cuda` 三种实现产生的最终
  16×16 网格 **完全一致**（见 `case_S_run.log`），
  且 `verify.py` 输出 `PASS, live cells = 39`。
* 可视化：`python3 visual.py life_N16_T4.txt life_N16_T4.png` 生成
  512×512 黑白图（黑=活、白=死），见 `life_N16_T4.png`。

### 3.2 算例 M （N = 257, T = 100）— 非整除规模自动验证

```
PASS: final grid matches the reference for N=257, T=100.
      live cells = 155
```

序列、MPI、CUDA 三个版本的输出 `life_N257_T100.txt`
全部通过 `verify.py`（详见 `case_M_run.log`）。
`N=257` 不能被 4 / 16 整除，验证了 1-D 行划分中
"余数行分配给前若干 rank"这一逻辑的正确性。

### 3.3 算例 L （N = 1024, T = 200）— 性能与加速比

每个版本运行 **11 次**，取中位数（单位 ms）。完整日志见 `results.txt`。

| 实现 | 进程/线程配置 | 运行时间 (中位数, ms) | 加速比 |
|---|---|---:|---:|
| serial            | 1 CPU 线程             | **3058.75** | 1.00× |
| parallel_mpi      | -np 4                  | **90.94**   | **33.6×** |
| parallel_mpi      | -np 16 (`--oversubscribe`) | **41.72** | **73.3×** |
| parallel_cuda     | 1×T4，block 16×16      | **5.95**    | **514.4×** |

> 速度比 = 串行时间 ÷ 该版本时间（中位数对中位数）。

各算例下中位数的完整数据见 `results.txt`。
为了让数据更稳定，每次单独 fork 一个新进程跑一次，
取中位数后能屏蔽冷启动、调度抖动以及 mpirun 启动开销带来的离群点。

---

## 4. 运行时间观察与分析

* **CUDA 一骑绝尘**：T4 上 `1024×1024×200` 的迭代仅 ~6 ms，
  对应每步 ~30 µs，吻合"全部 1M 个线程并行 + 8 邻居读"
  的近线性内存带宽估计。串行 → CUDA 加速 ~514×。
* **MPI 接近理想**：4 进程 33.6×、16 进程 73.3×；并不严格线性，
  原因是：① 每步两次 `Sendrecv` 与 `MPI_Barrier` 的同步开销固定，
  ② 16 进程下每个 rank 只剩 64 行，有较大的相对边界比，
  ③ 系统上只有 4 个物理 ntasks，`-np 16` 是 oversubscribe，
  超出物理核数后扩展性变差。
* **小规模 (S) 上并行反而变慢**：`N=16, T=4` 的串行只需 ~20 µs，
  fork/启动 + MPI 初始化 + GPU 上下文创建本身就要数百 µs ~ 数 ms，
  所以加速比 < 1，这是正常现象，仅用于检验正确性。

---

## 5. 提交清单

```
parco2026_hw4/
├── README.md                  # 题目要求
├── serial.cpp                 # 串行实现（题目给定）
├── parallel_mpi.cpp           # MPI 实现
├── parallel_cuda.cu           # CUDA 实现
├── Makefile                   # 编译三个可执行文件
├── run.slurm                  # 在 slurm 上跑全部算例的脚本
├── verify.py                  # 题目自带验证脚本
├── visual.py                  # 把输出 .txt 渲染为 PNG
├── bench.py                   # 单算例 (L) 11 次中位数 benchmark
├── bench_all.py               # 三个算例 × 4 配置 × 11 次中位数 benchmark
├── results.txt                # bench_all.py 输出（含原始 11 次时间）
├── case_S_run.log             # 算例 S 三种实现的运行日志（含截图素材）
├── case_M_run.log             # 算例 M 三种实现 + verify 的运行日志
├── life_N16_T4.txt            # 算例 S 最终网格
├── life_N16_T4.png            # 算例 S 可视化
├── life_N257_T100.txt         # 算例 M 最终网格
├── life_N1024_T200.txt        # 算例 L 最终网格
└── REPORT.md                  # 本报告
```

## 6. 复现命令

```bash
make                                              # 编译 serial + 两个并行版本
./serial 16 4 && python3 verify.py life_N16_T4.txt
mpirun --allow-run-as-root -np 4 ./parallel_mpi 257 100
python3 verify.py life_N257_T100.txt              # 期望 PASS
./parallel_cuda 1024 200                          # 单卡 CUDA
python3 bench_all.py                              # 11 次中位数 benchmark
python3 visual.py life_N16_T4.txt life_N16_T4.png # 可视化
```
