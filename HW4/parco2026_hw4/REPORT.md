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

时间测量方式：算例 L 下每个版本运行 **11 次**，记录每次内核计算时间
（串行/MPI 用 `std::chrono`，CUDA 在 `cudaDeviceSynchronize` 前后取时间），
取 **中位数** 作为最终运行时间，并以串行中位数为基准计算加速比。
算例 S、M 只跑一次用于正确性验证（`verify.py`）。整套流程已经直接
集成到 `run.slurm`，原始每次的时间和最终中位数都打印到 SLURM 的
`job_<id>_<node>.out` 日志中，不再写入单独的 `results.txt`。

---

## 2. 并行划分与边界处理

### 2.1 MPI 实现 (`parallel_mpi.cpp`)

* **划分方式**：1-D 行划分。当 `N` 不能被进程数 `P` 整除时，
  把余数 `N % P` 行依次分配到前 `N % P` 个进程，每个进程负责
  `local_rows = ⌊N/P⌋ + (rank < N%P)` 行。
* **本地存储**：每个进程持有一个 `(local_rows + 2) × (N + 2)` 的字节矩阵 —
  上下两行是 ghost rows，左右两列也额外补一列零（恒为 0 的 ghost cols）。
  这样内层 stencil 不需要再判断 `jl >= 0` / `jr < N`，循环体退化为
  纯 8 加约简，g++ `-O3` 能自动向量化。
* **每步流程**：
  1. `MPI_Sendrecv`：把自己的最上一行（行 `1` 的真实数据，从 col 1 起 N 字节）
     送给上邻居 `rank-1`，同时从下邻居接收下 ghost row。
  2. 再做一次 `MPI_Sendrecv` 把最下一行送给下邻居、从上邻居接收上 ghost row。
  3. 在全局上下边界 (`rank==0` / `rank==P-1`) 处把对应 ghost row 用 `memset(0)` 清零，
     从而严格满足"网格外格子全为 0"的固定死亡边界。
  4. 双重循环更新 `next` 中对应的 `local_rows` 行；左右 ghost cols 恒为 0
     直接参与累加，无需边界判断。
  5. `current.swap(next)`。
* **边界处理小结**：
  * 上下边界：通过 `MPI_PROC_NULL` 让 `MPI_Sendrecv` 在端点 rank 自动跳过通信，
    再 `memset` ghost row 为 0；
  * 左右边界：靠永远为 0 的 ghost cols 无分支地处理；
  * 三种边界条件都与串行版的 `live_neighbors` 保持完全一致。
* **结果汇集**：每个 rank 先把自己的 `local_rows × N` 真实数据从 padded 缓冲区
  打包到一个紧凑 buffer，再用 `MPI_Gatherv(counts, offsets)` 聚合到 rank 0，
  最后写入 `life_N<N>_T<T>.txt`。

### 2.2 CUDA 实现 (`parallel_cuda.cu`)

* **划分方式**：每个 CUDA 线程负责一个格子；线程块为 `32×4`
  （32 个线程恰好一个 warp，让 byte 级 global load 完美合并；4 行
  = 4 warps/block 在 T4 上保持占用率合适），网格大小向上取整为
  `(⌈N/32⌉, ⌈N/4⌉)`。
* **双缓冲**：在显存上分配 `d_current`、`d_next` 两块 `N×N` 字节缓冲；
  每步内核从 `d_current` 读、向 `d_next` 写，结束后在 host 端
  `std::swap(d_current, d_next)`。
* **边界**：内核里直接判断 `0 ≤ ni < N && 0 ≤ nj < N`，
  避免把网格外的"死格子"读进来。这样实现简洁，
  且对 `N=257` 这种非整除规模天然支持（多余线程在 `i>=N || j>=N`
  时早返回）。
* **显存访问**：网格按行主序连续存储，
  同一 warp 的 32 个连续线程读同一行的 32 个连续 byte，
  完美 coalesce。8 个邻居读会触发 L1 / 纹理缓存的局部命中。
* **测时与冷启动**：测时之前先做一次 warm-up 内核启动并
  `cudaDeviceSynchronize`，再把初始网格重新拷回设备，
  避免把 nvcc JIT/runtime 初始化开销算进 `T` 步迭代时间。
* **block size 选择**：尝试过 `8×8 / 8×16 / 8×32 / 16×8 / 16×16 / 16×32 /
  32×4 / 32×8 / 32×16` 共 9 种 power-of-two 块形（11-run 中位数），
  `32×4` 在 T4 上是清一色赢家（3.17 ms），其次是 `32×8`（3.95 ms）；
  `16×16`（5.34 ms，原默认值）只能进前五。决定性因素是 warp 与
  byte-coalesced 访存的对齐。

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

每个版本运行 **21 次**，取中位数（单位 ms）。原始每次运行的时间以及
中位数 / 加速比汇总都直接打印在 `run.slurm` 的 SLURM 日志
`job_<id>_<node>.out` 里（"Case L benchmark" 与 "Case L summary" 两节）。

**最新结果**（应用 M1 halo-padded 内核 + C5 32×4 block 后）：

| 实现 | 进程/线程配置 | 运行时间 (中位数, ms) | 加速比 |
|---|---|---:|---:|
| serial            | 1 CPU 线程             | **2711.58** | 1.00× |
| parallel_mpi      | -np 1                  | **330.75**  | 8.20× |
| parallel_mpi      | -np 4                  | **90.82**   | **29.86×** |
| parallel_mpi      | -np 8                  | **82.42**   | 32.90× |
| parallel_mpi      | -np 16 (`--oversubscribe`) | **41.31** | **65.63×** |
| parallel_cuda     | 1×T4，block 32×4       | **3.17**    | **854.22×** |

> 速度比 = 串行时间 ÷ 该版本时间（中位数对中位数）。

每次单独 fork 一个新进程跑一次，取中位数后能屏蔽冷启动、调度抖动以及
mpirun 启动开销带来的离群点。

#### 优化历程（commit-by-commit 探索）

提交记录按 "try → 通过 / revert" 模式串成一条审计线，
所有候选方向是否保留的依据写在 `BENCH_LOG.md` 里：

| 方向 | 描述 | Δ vs 上一基线 | 决策 |
|---|---|---:|---|
| **M1** | MPI inner buffer halo-pad 到 `(local_rows+2)×(N+2)`，去掉左右边界 if 分支 | mpi_np4 −11.0% | **保留** |
| M2 | MPI 改用 `Isend/Irecv` + 内层 row 与 halo 通信重叠 | mpi_np4 +2.7% | drop |
| M3 | Hybrid MPI + OpenMP，4 种 P×T 组合 | 最优 hybrid +4.6% vs mpi_np16 | drop |
| C1 | CUDA `__ldg()` read-only loads | cuda −0.4% | drop |
| C2 | CUDA 设备缓冲 halo-pad 到 `(N+2)×(N+2)` | cuda +2.4% | drop |
| C3 | CUDA shared-memory tiling，`(BX+2)×(BY+2)` halo 一次加载 | cuda +57% | drop |
| **C5** | CUDA block size 16×16 → 32×4（一个 warp 完整覆盖一行） | cuda −39.6% | **保留** |
| QoL | `write_grid` 用一次 `ofs.write` 替换逐字节 `<<` | I/O ~8× | 保留（非计算时间） |

最终保留的两项优化分别命中了两条不同的瓶颈：

* **M1（halo-pad）** 干掉了 MPI 内核里每个格子的 `jl/jr` 边界判断，
  使内层 8 加约简能被 g++ 自动向量化（更友好的内存连续访问）。在
  `mpi_np4` 上 −11%；`mpi_np16` 上每个 rank 只有 64 行，本身已经
  通信受限，提升只有 −1.3%。
* **C5（32×4 block）** 把 16×16 改成 32×4：32 个线程恰好一个 warp，
  对 byte 数据的全局 load 100% 合并；4 行/block = 4 warps/block 在 T4 上
  保持占用率合适。这是单 GPU 最大的一次提升 — cuda 5.3 → 3.2 ms。
  这个 block 形状对 N 是 32 的整数倍时最有利，但即使是 N=257 也是
  32+1 行，多余线程会被 `(i>=N || j>=N)` 早返回，正确性不受影响。

被丢弃的方向都有共同主题：在已经"够小够紧"的内核或计算密集场景里，
* M2 的非阻塞通信在单节点 vader 共享内存 BTL 上没有可掩盖的通信窗口；
* M3 的 OpenMP fork/join 在 200 步小循环里的开销超过了通信节省；
* C1 的 `__ldg` 在 nvcc/sm_60 已经对 `const __restrict__` 默认走只读 cache 的情况下没有额外作用；
* C2 的 device 端 halo pad 增加了一行 4 byte stride 与额外的 `cudaMemcpy`，得不偿失；
* C3 的 shared-memory tiling 在内核已经如此短的情况下，`__syncthreads` 与边界 halo 加载分支带来的代价超过了访存节约。

完整探索过程写在 `BENCH_LOG.md`，每个 try / revert 都有对应的 git commit。

---

## 4. 运行时间观察与分析

* **CUDA 一骑绝尘**：T4 上 `1024×1024×200` 的迭代仅 ~3.17 ms，
  对应每步 ~16 µs，几乎贴着 T4 的 GDDR6 带宽上限。
  串行 → CUDA 加速 ~854×。32×4 block 比常见的 16×16 默认值快 ~40%，
  纯靠让 warp 与 byte-coalesced load 对齐。
* **MPI 接近理想**：4 进程 ~30×、16 进程 ~66×；并不严格线性，
  原因是：① 每步两次 `Sendrecv` 与 `MPI_Barrier` 的同步开销固定，
  ② 16 进程下每个 rank 只剩 64 行，有较大的相对边界比，
  ③ 系统上只有 4 个物理 ntasks，`-np 16` 是 oversubscribe，
  超出物理核数后扩展性变差。M1 的 halo-pad 内核改进让 `mpi_np4`
  得到 −11% 的提升，但 `mpi_np16` 因为已被通信 / barrier 锁死，
  只能挤出 −1%。
* **小规模 (S) 上并行反而变慢**：`N=16, T=4` 的串行只需 ~20 µs，
  fork/启动 + MPI 初始化 + GPU 上下文创建本身就要数百 µs ~ 数 ms，
  所以加速比 < 1，这是正常现象，仅用于检验正确性。
* **被丢弃的方向**：M2（非阻塞通信 + 内层 row 重叠）、M3（hybrid
  MPI+OpenMP）、C1（`__ldg`）、C2（CUDA halo-pad）、C3（CUDA
  shared-memory tiling）都没有跑赢已有基线 ≥10% 的门槛。原因写在
  `BENCH_LOG.md` 与对应 commit message 里 — 这些都是工程上"看起来
  应该有用"但被实际数据否决的优化，记录它们也是这次实验的一部分。

---

## 5. 提交清单

```
parco2026_hw4/
├── README.md                  # 题目要求
├── serial.cpp                 # 串行实现（题目给定 + buffered I/O）
├── parallel_mpi.cpp           # MPI 实现（halo-padded 内核）
├── parallel_cuda.cu           # CUDA 实现（32×4 block）
├── Makefile                   # 编译三个可执行文件
├── run.slurm                  # 在 slurm 上跑全部算例 + 算例 L benchmark 的脚本
├── verify.py                  # 题目自带验证脚本
├── visual.py                  # 把输出 .txt 渲染为 PNG
├── bench.py                   # 单算例 (L) 11 次中位数 benchmark（旧 Python 版，已不再被 run.slurm 调用）
├── BENCH_LOG.md               # 优化实验的 try/decision 记录（每条对应一个 commit）
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
make                  # 编译 serial + 两个并行版本
sbatch run.slurm      # 在 slurm 上一次跑完 S/M 验证 + L benchmark；输出落在 job_<id>_<node>.out
bash   run.slurm      # 等价的本地复现：#SBATCH 行只是注释，bash 会忽略
```
