# 并行计算 II · 作业 3 报告：CUDA 并行前缀和

## 1. 实验概述

实现了一维数组的并行 inclusive 前缀和（scan）。输入数组按公式
`x[i] = (i*17 + 13) % 1000` 生成，输出为 `y[i] = sum_{j=0..i} x[j]`，
长度均为 `N`。

代码文件：

| 文件 | 说明 |
|---|---|
| `serial.cpp`   | 题目提供的串行参考实现，未修改。 |
| `parallel.cu`  | CUDA 实现（本作业核心），不依赖 Thrust/CUB/cuBLAS。 |
| `Makefile`     | 同时编译 `serial` 和 `parallel`。 |
| `verify.py`    | 题目提供的验证脚本，未修改。 |
| `run.slurm`    | 集群批处理脚本（编译、跑 S/M/L、串行与并行各 11 次取中位数、计算加速比）。 |

本地实验环境（用于开发与正式测量）：

* GPU：NVIDIA Tesla T4（Compute Capability 7.5，16 GB GDDR6），
  本次测量时 GPU **空闲**（`nvidia-smi` 显示 0% 利用率，2 MiB 显存占用）。
* CUDA：12.1（`nvcc` V12.1.105），驱动 535.161.07。
* 主机端：单线程串行 `g++ -O2 -std=c++11`。
* 操作系统：Linux x86_64。

> 测量方法：算例 L 的串行与并行 **均运行 11 次**，并取中位数作为代表值；
> 并行版本另跑 1 次预热（不计时）以排除 CUDA context 首次初始化的影响。

## 2. 算法思路

题目建议「块内 scan + 块和 scan + 偏移回加」三段式，本实现采用了
经典的 **Blelloch 上扫/下扫（work-efficient prefix sum）** 在共享内存
中完成块内 scan，再用同一套 kernel 链 **递归** 地对每一层块和数组继续
做 scan，最后把每个块的 *exclusive* 前缀和作为偏移加回去。

### 2.1 三段式三个 kernel

设每 block 处理 `ELEMENTS_PER_BLOCK = 2 * BLOCK_SIZE` 个元素，每个
线程负责其中两个元素（这是 Blelloch 算法常见的布局：`n` 个元素只需
`n/2` 个线程即可完成完整树）。

1. **`block_scan_kernel`**：块内 scan
   * 把 `2*BLOCK_SIZE` 个元素加载到共享内存 `sdata`。尾块用 0 填充
     越界元素，使所有 block 走一致的代码路径，无需特例分支。
   * **Up-sweep（reduce）**：自底向上把 `sdata` 当作完全二叉树，
     `log2(2*BLOCK_SIZE)` 次同步，每一层让活跃线程做一次加法。
   * `sdata[2*BLOCK_SIZE-1]` 此刻是块总和，把它写到
     `block_sums[blockIdx.x]`，并把该位置清零（这是 Blelloch 算法
     从 inclusive 转 exclusive 的标准技巧）。
   * **Down-sweep**：自顶向下完成 exclusive scan。
   * 最后把当前线程的 *原始值* 加回到 exclusive scan 上，得到块内
     **inclusive** scan，写回全局内存 `y`。

2. **递归 scan：`inclusive_scan_recursive(d_block_sums)`**
   * 对长度为 `num_blocks` 的块和数组再调一次相同的三段式流程。
   * 当数组规模 `M ≤ ELEMENTS_PER_BLOCK` 时，一个 block 即可全部
     处理完，递归终止；这是天然的 base case。
   * 由 `block_scan_kernel` 给出的 `d_block_inclusive` 是 inclusive 的，
     而我们需要的是 exclusive（block 0 的偏移应为 0），用一个轻量
     kernel `incl_to_excl_kernel` 做 `excl[i] = incl[i] - original[i]`
     转换。这避免了写一个独立的 exclusive 版本 kernel。

3. **`add_block_offsets_kernel`**：把 exclusive 的块偏移加回每个元素。
   * 每个 block 的所有 `2*BLOCK_SIZE` 个元素加上 `block_offsets[blockIdx.x]`。

### 2.2 几个工程细节

* **类型**：累加用 `long long`，`N=8388608` 的总和约 4.19e9，超出
  `int32` 范围；与 `serial.cpp` 一致。
* **输入生成**：在 GPU 端用 `generate_input_kernel` 直接生成 `d_x`，
  避免把 `N` 个 long long 从主机拷贝到设备（对于 8M 元素这本身就要
  ~64 MB 的 H2D 拷贝，对加速比测量不公平也没必要）。
* **越界处理**：`block_scan_kernel` 对越界 lane 加载 `0`，写回时
  通过 `if (idx < N)` 守门。算法对任意 `N`（含 `N=1000003` 这种
  非 2 的幂）都成立——这是题目算例 M 专门考察的点。
* **正确性参考实现**：`serial` 与 `parallel` 同用文件名
  `scan_N<N>.txt`，所以同一目录下后跑的会覆盖前者。`run.slurm`
  里两个程序都跑，`verify.py` 验证的就是 *parallel* 的输出。
* **不依赖任何并行库**：仅使用 CUDA C/C++、`cudaEvent`、共享内存、
  全局内存，无 Thrust/CUB/cuBLAS。

### 2.3 复杂度

* 总加法次数：`O(N)`（work-efficient，与串行同阶）。
* 步长：`O(log N)`（递归层数 `log_{2*BLOCK_SIZE}(N)`，对 `N=8M、
  BLOCK_SIZE=512` 仅 2 层）。
* 共享内存：每 block `2*BLOCK_SIZE * sizeof(long long)`，
  `BLOCK_SIZE=512` 时 8 KB，远低于 T4 的 64 KB/SM 共享内存上限。

## 3. Block 大小选择

在 **空闲** Tesla T4 上对 `N=8388608` 做 BLOCK_SIZE 扫描（每个配置
预热一次，之后跑 11 次，取中位数；只比较 kernel-only 的 scan 时间）：

| BLOCK_SIZE | ELEMENTS_PER_BLOCK | 共享内存/block | scan 中位数 (ms) |
|---:|---:|---:|---:|
| 128  | 256  | 2 KB  | 1.797 |
| 256  | 512  | 4 KB  | **1.726** |
| 512  | 1024 | 8 KB  | 1.761 |
| 1024 | 2048 | 16 KB | 1.909 |

观察：

* `BLOCK_SIZE=256` 在 T4 上最快，但 128/512 都在 ~1% 以内，可视作
  并列。`1024` 略慢约 8%。
* 原因：T4 SM 上每 block 寄存器与共享内存占用越小，每个 SM 能驻留
  越多 block，越能掩盖访存延迟（occupancy 提升）。`BLOCK_SIZE=1024`
  时块数最少，且每块同步成本最高，wave 数下降导致尾部利用率受损。
* 整体差异在 ~10% 以内，对正确性无影响。

最终选择 **`BLOCK_SIZE = 512`**：在 T4 上中位数仅比最佳的 256 慢 ~2%，
并且：
1. 在数据规模更大、SM 共享内存更宽裕的高端卡（V100/A100）上通常更
   平衡；
2. 减少了递归层数（`N=8M` 时 `BLOCK_SIZE=128` 要 3 层、512 只要 2 层），
   降低 kernel 启动开销在小规模 `N` 时的占比。

在 `parallel.cu` 里这是一个 `static const` 常量，集群上想换值只需要
改一行重编译。

## 4. 三个算例的运行结果

### 4.1 算例 S：`N = 32`（正确性 / 截图）

**串行**：

```
$ ./serial 32
Executing serial prefix sum with N = 32
Serial compute time: 0.00018 ms
Wrote result to scan_N32.txt
y = 13 43 90 154 235 333 448 580 729 895 1078 1278 1495 1729 1980 2248 \
    2533 2835 3154 3490 3843 4213 4600 5004 5425 5863 6318 6790 7279 7785 8308 8848
```

**CUDA**（GPU 空闲时实测）：

```
$ ./parallel 32
Executing CUDA prefix sum with N = 32
GPU: Tesla T4 (CC 7.5)
BLOCK_SIZE = 512, ELEMENTS_PER_BLOCK = 1024
CUDA scan (kernels only): 0.044 ms
CUDA total (gen+scan+D2H): 0.383 ms
Wrote result to scan_N32.txt
y = 13 43 90 154 235 333 448 580 729 895 1078 1278 1495 1729 1980 2248 \
    2533 2835 3154 3490 3843 4213 4600 5004 5425 5863 6318 6790 7279 7785 8308 8848
```

`verify.py scan_N32.txt` ⇒ **PASS: all 32 elements match the reference prefix sum.**
`y[0]=13, y[31]=8848`，与串行输出逐元素一致。

> 说明：`N=32` 时一次 kernel 启动+同步约 0.04 ms，已可见但仍被
> 主机端 `cudaEventSynchronize` 等固定开销主导；这一规模下 CPU 反而
> 更快（参考时间 0.18 µs），真正体现加速比的是算例 L。

### 4.2 算例 M：`N = 1000003`（非 2 的幂，自动验证）

```
$ ./parallel 1000003
Executing CUDA prefix sum with N = 1000003
GPU: Tesla T4 (CC 7.5)
BLOCK_SIZE = 512, ELEMENTS_PER_BLOCK = 1024
CUDA scan (kernels only): 0.422 ms
CUDA total (gen+scan+D2H): 3.77 ms
Wrote result to scan_N1000003.txt

$ python3 verify.py scan_N1000003.txt
PASS: all 1000003 elements match the reference prefix sum.
      y[0] = 13, y[-1] = 499500090
```

* `N = 1000003` 是质数，远非 2 的幂，且不能被 `ELEMENTS_PER_BLOCK = 1024` 整除
  （余 707）。零填充 + `if (idx < N)` 守门让尾块同样正确。
* 验证脚本的 `PASS` 行同时打印 `y[0]` 和 `y[N-1]`，与 `serial 1000003` 输出一致。

### 4.3 算例 L：`N = 8388608`（性能 / 加速比）

**测量协议**：串行与并行 **均运行 11 次**，取中位数。并行另跑一次预热
（不计时）以排除 CUDA context 首次初始化。GPU 在测量期间空闲。

#### 11 次原始计时（按运行顺序）

```
Serial run  1: 20.9551 ms      CUDA run  1: scan=1.80605 ms, total=26.4552 ms
Serial run  2: 21.5369 ms      CUDA run  2: scan=1.76534 ms, total=26.7215 ms
Serial run  3: 20.8042 ms      CUDA run  3: scan=2.05773 ms, total=26.1371 ms
Serial run  4: 20.7672 ms      CUDA run  4: scan=2.03366 ms, total=26.3488 ms
Serial run  5: 21.9012 ms      CUDA run  5: scan=2.10330 ms, total=26.3008 ms
Serial run  6: 21.0586 ms      CUDA run  6: scan=1.73466 ms, total=26.8890 ms
Serial run  7: 20.9094 ms      CUDA run  7: scan=1.71776 ms, total=26.1617 ms
Serial run  8: 20.9540 ms      CUDA run  8: scan=1.92707 ms, total=26.5626 ms
Serial run  9: 21.8622 ms      CUDA run  9: scan=1.70826 ms, total=26.5779 ms
Serial run 10: 21.3658 ms      CUDA run 10: scan=1.76128 ms, total=26.6372 ms
Serial run 11: 20.9479 ms      CUDA run 11: scan=1.74538 ms, total=26.7145 ms
```

#### 统计汇总

| 指标 | min | **median** | max |
|---|---:|---:|---:|
| Serial compute time（ms）             | 20.767 | **20.955** | 21.901 |
| CUDA scan kernels only（ms）          |  1.708 |  **1.765** |  2.103 |
| CUDA total = 输入生成+scan+D2H（ms）   |        | **26.563** |        |

| 加速比 | 公式 | 数值 |
|---|---|---:|
| **加速比（scan only）** | 20.955 / 1.765 | **≈ 11.87×** |
| 加速比（含 D2H 拷贝）   | 20.955 / 26.563 | ≈ 0.79× |

`verify.py scan_N8388608.txt` ⇒ **PASS: all 8388608 elements match the reference prefix sum.**
（`y[N-1] = 4190102880`，超出 `int32` 范围，因此用 `long long` 累加是必要的。）

#### 关于「含 D2H」的解读

* 串行测量的是 *算法* 时间，不包括把 64 MB 结果写文件的 I/O。
* CUDA 的 `total` 包括 GPU 端生成输入 + 三段式 scan + 8M long long 的
  D2H 拷贝（≈ 64 MB）。
  - 在 PCIe Gen3 ×16 上理论上限约 12 GB/s，64 MB 的下限约 5–6 ms；
  - 实测 D2H ≈ 26.5 - 1.8 ≈ **24.7 ms**，远高于理论值——这是因为该
    虚拟机的 GPU 通过共享 PCIe 通道挂载，可用带宽仅约 2.6 GB/s。
    在物理直连 PCIe Gen3/Gen4 的集群节点上，这一数字会显著降低。
* 因此 **「scan only」对应公平的算法对比，约 11.87× 加速比**；
  「含 D2H」反映端到端的实际性能，主要被 PCIe 带宽限制，且强烈依赖
  集群的硬件配置。
* scan 计算本身是 *访存密集* 而非计算密集（每个元素只做 1 次加法）：
  1.77 ms 处理 64 MB（读 + 写）≈ 70 GB/s 有效全局带宽，是 T4 理论
  带宽 320 GB/s 的 ~22%；进一步压榨需要 **decoupled look-back** 单遍
  scan（Merrill & Garland 2016，CUB 内部用法）或将累加类型改为 `int`，
  但都超出本作业要求。

## 5. 运行时间观察与小结

* 三段式 + 共享内存内的 Blelloch 树是一个 **work-efficient** 算法
  （线性 work，对数 step），实测在 T4 上对 8M long long 输入达到
  **~11.87×** 加速比，与典型 CUDA scan 教科书示例（GPU Gems 3 第 39 章）
  的量级吻合。
* `N` 越小，CUDA 启动/同步开销占比越大：
  - `N=32` 时 kernel scan 仅 0.04 ms，但 CPU 串行用 0.0002 ms，
    GPU **慢于 CPU**；
  - `N=10^6` 时 kernel scan 已经压到 0.42 ms，串行 ≈ 2.5 ms，
    scan-only 加速比 ~6×；
  - `N=8M` 时 scan-only 加速比攀升到 ~12×。
* `N` 越大，GPU 占满 SM、kernel 占比上升，加速比逼近 GPU 与 CPU 的
  带宽比；继续增大 `N` 主要靠提升带宽利用率获得更高加速比。
* 串行版本本身也受 CPU 缓存影响：连续 11 次串行 8M 数据的范围在
  20.77–21.90 ms 之间（极差 ~5%），中位数 20.96 ms 比 best-of-N
  更稳健，这也是本次将统计量统一改成中位数的原因。
* 选择合适的 `BLOCK_SIZE` 对最终性能影响在 T4 上约 10%，最佳值
  与卡相关；脚本里把它做成了 `static const`，便于一行替换重编译。

## 6. 文件清单

```
parco2026_hw3/
├── Makefile          # 编译 serial 和 parallel
├── README.md         # 题目原文
├── parallel.cu       # CUDA 实现
├── serial.cpp        # 题目自带串行实现
├── verify.py         # 题目自带验证脚本
├── run.slurm         # 集群批处理脚本：完整跑 S/M/L 并打印加速比
└── REPORT.md         # 本报告
```

## 7. 集群上复现

直接在仓库目录提交即可：

```bash
sbatch run.slurm
```

`run.slurm` 会做：

1. 加载 CUDA 模块（如有），运行 `make`；
2. 跑算例 S、M、L 并各自调用 `verify.py`；
3. 对算例 L **串行重复 11 次、并行预热 1 次后再重复 11 次**，分别取
   **中位数**，最后输出
   `Serial median`、`CUDA median scan`、`CUDA median total`、
   `Speedup (scan only)` 和 `Speedup (incl. D2H)` 的最终摘要，
   同时附带 min/max 用于判断稳定性。

输出会写到 `job_<JOBID>_<NODE>.out`，里面同时包含 `N=32` 的运行回显
（截图用）、`N=1000003` 的 `PASS` 行（截图用）以及 `N=8388608` 的
完整时间表与加速比。
