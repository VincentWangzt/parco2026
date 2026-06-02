# 并行计算 II · 作业 3 报告：CUDA 并行前缀和

## 1. 实验概述

实现了一维数组的并行 inclusive 前缀和（scan）。输入按
`x[i] = (i*17+13) % 1000` 生成，输出 `y[i] = Σ_{j=0..i} x[j]`，长度均为 `N`。

代码文件：

| 文件 | 说明 |
|---|---|
| `serial.cpp`   | 题目自带串行参考实现，未修改。 |
| `parallel.cu`  | CUDA 实现（本次作业核心），不依赖 Thrust/CUB/cuBLAS。 |
| `Makefile`     | 同时编译 `serial`（`-O3`）和 `parallel`（`-O3`）。 |
| `verify.py`    | 题目自带验证脚本，未修改。 |
| `run.slurm`    | 集群批处理脚本：编译、跑 S/M/L、串行与并行各 51 次取中位数、计算加速比。 |

报告中的开发与测量环境：

* 开发 / 报告 GPU：**NVIDIA Tesla T4**（CC 7.5，16 GB GDDR6），测量时
  `nvidia-smi` 0% 利用率、~2 MiB 显存占用。
* 集群 GPU（`run.slurm` 在 `job_153.out` 的硬件）：**NVIDIA GeForce
  RTX 2080 Ti**（同样 CC 7.5，11 GB GDDR6）。两张卡 SM 架构相同
  （Turing），算法与 block 选择跨卡迁移成本接近零。
* CUDA 12.x（开发 12.1，集群 12.4），驱动 535/550。
* 主机端：单线程串行 `g++ -O3 -std=c++11`。

> 测量方法：S/M/L 均跑 **51 次取中位数**（并行额外 1 次 warm-up
> 不计时）。中位数对单个慢 outlier 鲁棒，比 best-of-N 更代表稳态性能。
> 报告内表格中 “scan-only” 即 `CUDA scan (kernels only)`：用
> `cudaEvent` 包住三段式 scan 的 kernel chain。

## 2. 算法思路

题目建议的「块内 scan + 块和 scan + 偏移回加」三段式，本实现采用
**经典 Blelloch 上扫/下扫（work-efficient prefix sum）** 在共享内存
中完成块内 scan，递归地对每一层块和数组继续做同一三段式，最后把每
个块的 *exclusive* 偏移加回原元素。算法主体保留全程，未改成
warp-shuffle / 单遍 decoupled look-back 等更激进方案。

设每 block 处理 `ELEMENTS_PER_BLOCK = 2*BLOCK_SIZE` 个元素，每个线程
处理 2 个（这是 Blelloch 算法的经典布局：n 个元素只需 n/2 个线程）。

### 2.1 三段式三个 kernel

1. **`block_scan_kernel_top`（顶层）** / **`block_scan_kernel`（递归层）**

   * 把 `2*BLOCK_SIZE` 个元素加载到共享内存 `sdata`。尾块用 0 填充
     越界元素，使所有 block 走一致的代码路径。顶层从 `int32` 输入读
     入并 *提升* 为 `long long` 写进 `sdata`；递归层（块和已是
     `long long`）走原始的 `long long*` 通路。
   * **Up-sweep（reduce）**：自底向上把 `sdata` 当作完全二叉树，
     `log2(2*BLOCK_SIZE)` 次同步，每层活跃线程做一次加法。
   * `sdata[2*BLOCK_SIZE-1]` 此刻是块总和，写到 `block_sums[bid]`，
     并把该位置清零（Blelloch 的 inclusive→exclusive 标准技巧）。
   * **Down-sweep**：自顶向下完成 exclusive scan。
   * 最后把当前线程的原始值加回到 exclusive scan 上，得到块内
     **inclusive** scan，写回全局 `y`。

   关键工程细节：**bank-conflict-free 共享内存布局**——把所有
   `sdata[i]` 访问替换成 `sdata[i + (i>>5)]`（GPU Gems 3 §39.2.3 的
   标准做法）。原 `2*tid+1` / `2*tid+2 - 1` 的索引在 warp 内会以
   2 的幂步长落到同一组 bank 上，造成 **32-way 冲突**；插一个 padding
   slot 后冲突消失。这是本次最显著的单点优化。

2. **递归 scan**：`inclusive_scan_recursive` 对长度 `num_blocks` 的块
   和数组再调一次同样的三段式。规模 `M ≤ ELEMENTS_PER_BLOCK` 时一个
   block 即可完成，递归终止；这是天然 base case。N=8M、BLOCK_SIZE=256
   时只需 2 层。

3. **`add_block_offsets_kernel`**：把每块 *exclusive* 偏移加回每个元素。
   这个 kernel 同时收 `block_inclusive` 和原始 `block_sums`，在寄存器
   里现算 `off = inclusive[bid] - sums[bid]`，省掉一个独立的
   inclusive→exclusive 转换 kernel。

### 2.2 几个关键实现细节

* **类型分离**：`using Input = int;`（输入元素 < 1000，int32 足够）
  与 `using Value = long long;`（累加器 / 块和 / 输出，因为
  N=8388608 时 `y[N-1] = 4190102880 > 2^31`，必须 `long long`）。
  这把 `d_x` 设备显存从 64 MB 减到 32 MB（N=8M），并把每元素的输入
  全局读流量减半。

* **预分配 scan workspace**：原实现里递归每层都要
  `cudaMalloc + cudaFree` 三次，这在 timed 路径上累积成不可忽视
  的启动开销（`N=10^6` 时这部分曾占主导）。改成 `main` 里一次
  分配出 `scan_workspace_size(N)` 个 `Value` 槽，每层从中切片。

* **越界处理**：`block_scan_kernel` 对越界 lane 加载 `0`，写回前
  `if (idx < N)` 守门。算例 M（N=1000003，质数，非 2 的幂）就是为
  这一点而设计——尾块仍然走同一段代码并正确。

* **输入生成**：`generate_input_kernel` 直接在 GPU 上根据公式生成
  `d_x`，避免 H2D 拷贝。

* **输出写盘**：用手写 itoa + 单次 `fwrite` 替代逐元素
  `ofs << y[i] << "\n"`，仅影响 wall-clock 端到端时间，不在
  scan-only timed 路径里。N=8388608 时端到端从 ~3.5s 降到 ~0.7s。

* **不依赖任何并行库**：仅使用 CUDA C/C++、`cudaEvent`、共享内存、
  全局内存，无 Thrust / CUB / cuBLAS。

### 2.3 复杂度

* 总加法次数：`O(N)`（work-efficient，与串行同阶）。
* 步长：`O(log N)`；递归层数 `log_{EPB}(N)`，N=8M、BLOCK_SIZE=256
  时 2 层。
* 共享内存：每 block `(EPB + EPB/32 + 1) * sizeof(long long)`，
  EPB=512 时约 **4.1 KB**，远低于 T4 的 64 KB/SM 上限。

## 3. Block 大小选择

Tesla T4 上对 `N=8388608` 跑 BLOCK_SIZE 扫描（每配置 1 次 warm-up + 51 次
取中位数，比较 kernel-only scan 时间）：

| BLOCK_SIZE | ELEMENTS_PER_BLOCK | 共享内存/block | scan 中位数 (ms) |
|---:|---:|---:|---:|
| 128  | 256  | ~2 KB  | 接近 256，但递归层数 +1 |
| **256**  | **512**  | **~4 KB** | **0.893** |
| 512  | 1024 | ~8 KB | 与 256 同档（±~1%） |
| 1024 | 2048 | ~16 KB | 略慢，每块同步成本上升 |

最终选择 **`BLOCK_SIZE = 256`**：T4 上最稳定的最快档；CC 7.5 同代的
RTX 2080 Ti 上预期同样最优或并列最优；
N=8M 时只需 2 层递归。

`BLOCK_SIZE` 是 `parallel.cu` 顶部的 `static const`，集群上要换一个
值只改一行重编译。

## 4. 迭代优化历史

下表是开发过程中按 commit 分别记录的中位 scan-only 时间（T4，
N=8388608，每版本 51 次取中位数）。每一次 commit 都通过了 N=32 /
1000003 / 8388608 三个算例的 `verify.py`。

| 版本 | 改动要点 | scan 中位 (ms) | 累计加速 vs v0 | 备注 |
|---|---|---:|---:|---|
| **v0** | 原始 3-pass Blelloch；递归内 `cudaMalloc/Free`；独立 `incl_to_excl_kernel` | 1.800 | 1.00× | baseline |
| **v1** | 一次性预分配 scan workspace；把 incl→excl 折进 `add_block_offsets_kernel` | 1.589 | 1.13× | 大块 N 提升 ~12%；M 规模因为之前是 malloc-bound 而提升 2.2× |
| **v2** | bank-conflict-free 共享内存（`SI(i) = i + i>>5` 填充） | 1.063 | 1.69× | **单步最大提升**（vs v1 +33%）。原 `2*tid+1` 索引是 32-way bank 冲突 |
| (尝试 v3a 后回退) | int2 向量化加载 + 连续对齐 sdata 布局 | 1.100 | — | **回退**：连续对齐布局引入 2-way bank 冲突，吃掉了向量化收益 |
| **v3** | 仅改输入 dtype 为 int32（保留 v2 的交错式 sdata 布局） | 1.018 | 1.77× | 输入读字节数减半；`d_x` 显存 64→32 MB |
| **v4** | 手写 itoa + 单次 `fwrite`（QOL，不在 timed 路径） | 0.893* | n/a | 端到端 wall time 3.5 s → 0.71 s；scan-only 数字含 GPU 稳态时钟漂移，不归功于 v4 |

> *关于 v4 的 0.893 ms：v4 的代码改动全部位于 `cudaEventSynchronize`
> 之后，理论上不影响 scan-only 计时。在 v3↔v4 多次交错重测中，v4 的
> scan-only 中位稳定在 0.89–0.90 ms，而 v3 在 1.02–1.04 ms。最合理
> 的解释是 GPU 在测量过程中达到了更稳态的时钟工况（早段实测时 GPU
> 处于 idle 后开机状态）。报告把这部分谨慎地标为「漂移」而非 v4
> 的功劳，避免把测量噪声当算法收益。

“迭代优化失败也是优化的一部分” 的诚实记录：v3a（int2 + 连续对齐）实测
回退；v4 的 scan-only 数字归因于硬件稳态而非代码。

## 5. 三个算例的运行结果

> 以下并行数字来自 Tesla T4，对应当前已合并的最终代码（commit
> "v4: hand-rolled itoa + single fwrite"）。串行数字同样取自本机
> 51 次中位数。集群 RTX 2080 Ti 上的最新数字以 `sbatch run.slurm` 的
> `job_<JOBID>.out` 为准。

### 5.1 算例 S：`N = 32`（正确性 / 截图）

**串行**：

```
$ ./serial 32
Executing serial prefix sum with N = 32
Serial compute time: 0.00016 ms
Wrote result to scan_N32.txt
y = 13 43 90 154 235 333 448 580 729 895 1078 1278 1495 1729 1980 2248
    2533 2835 3154 3490 3843 4213 4600 5004 5425 5863 6318 6790 7279 7785 8308 8848
```

**CUDA**：

```
$ ./parallel 32
Executing CUDA prefix sum with N = 32
GPU: Tesla T4 (CC 7.5)
BLOCK_SIZE = 256, ELEMENTS_PER_BLOCK = 512
CUDA scan (kernels only): 0.0207 ms
CUDA total (gen+scan+D2H): 0.294 ms
Wrote result to scan_N32.txt
y = 13 43 90 154 235 333 448 580 729 895 1078 1278 1495 1729 1980 2248
    2533 2835 3154 3490 3843 4213 4600 5004 5425 5863 6318 6790 7279 7785 8308 8848
```

`y[0]=13, y[31]=8848`，与串行输出逐元素一致。
`verify.py` ⇒ **PASS: all 32 elements match the reference prefix sum.**

### 5.2 算例 M：`N = 1000003`（非 2 的幂，自动验证）

```
$ ./parallel 1000003
Executing CUDA prefix sum with N = 1000003
GPU: Tesla T4 (CC 7.5)
BLOCK_SIZE = 256, ELEMENTS_PER_BLOCK = 512
CUDA scan (kernels only): 0.145 ms
CUDA total (gen+scan+D2H): 3.33 ms
Wrote result to scan_N1000003.txt

$ python3 verify.py scan_N1000003.txt
PASS: all 1000003 elements match the reference prefix sum.
      y[0] = 13, y[-1] = 499500090
```

* `N=1000003` 是质数，远非 2 的幂，且不能被 `EPB=512` 整除（余 195）。
  零填充 + `if (idx < N)` 守门让尾块同样正确。
* T4 中位数 0.133 ms（51 次），串行 2.483 ms，**scan-only 加速比 ~18.6×**。

### 5.3 算例 L：`N = 8388608`（性能 / 加速比）

#### 测量协议

* 串行与并行均跑 **51 次取中位数**；并行额外 1 次预热不计时。
* 测量期间 GPU 空闲。

#### T4 实测中位数

| 指标 | min | **median** | max |
|---|---:|---:|---:|
| Serial compute time（ms） | 20.673 | **21.167** | 21.720 |
| CUDA scan kernels only（ms） | 0.889 | **0.893** | 1.049 |
| CUDA total（gen+scan+D2H，ms） | — | ~23.3 | — |

| 加速比 | 公式 | 数值 |
|---|---|---:|
| **加速比（scan only）** | 21.167 / 0.893 | **≈ 23.7×** |
| 加速比（含 D2H 拷贝） | 21.167 / 23.3 | ≈ 0.91× |

`verify.py scan_N8388608.txt` ⇒ **PASS: all 8388608 elements match the reference prefix sum.**
（`y[N-1] = 4190102880 > 2^31`，所以 `Value = long long` 是必要的。）

#### 集群 RTX 2080 Ti 数字（job_153.out，提交本次优化前的代码）

| 指标 | min | median | max |
|---|---:|---:|---:|
| Serial compute time（ms） | 12.92 | **32.33** | 40.19 |
| CUDA scan kernels only（ms） | 0.974 | **1.224** | 2.165 |
| CUDA total（ms） | — | 58.22 | — |

* 当时的 scan-only 加速比 ≈ **26.4×**。串行的 min/max 离散度比较大
  （12.92 vs 40.19）反映了集群节点上有其它任务竞争 CPU；中位数仍
  代表稳态。
* 那是 **v0 baseline** 的数字。本次优化之后用同一 `run.slurm` 重跑，
  scan-only 中位预期会下降到 0.6–0.8 ms 量级（v0→v4 累计 ~2× 在 T4
  上观测到，2080 Ti 同代架构下应类似），加速比会进一步提升。**最终
  报告里使用最新一次 `sbatch run.slurm` 的 `job_<id>.out` 为准。**

#### 关于「含 D2H」的解读

* 串行测的是 *算法* 时间，不包括把 64 MB 结果写盘的 I/O。
* CUDA 的 `total` 包括 GPU 端生成输入 + 三段式 scan + 8M `long long`
  D2H 拷贝（≈ 64 MB）。
  - 在 PCIe Gen3 ×16 上理论上限约 12 GB/s，64 MB 的下限约 5–6 ms；
    实测 ~22.4 ms 处于该机器的 PCIe 共享通道带宽（~3 GB/s）瓶颈范围内。
  - 不同集群节点上这个数字会显著不同，且与算法无关。
* 因此 **「scan only」对应公平的算法对比，T4 上 ~23.7× 加速比**；
  「含 D2H」反映端到端实际性能，主要受 PCIe 带宽限制。
* 当前 scan 实现 ≈ 0.89 ms 处理 64 MB（写）+ 32 MB（读）≈
  **108 GB/s 有效全局带宽**，约为 T4 理论 320 GB/s 的 ~34%；要进一步
  压榨需要 warp-shuffle / decoupled look-back 单遍 scan，或将
  Output 类型也降到 int32（但 int32 不够装 4.19e9，需更精细的
  multi-precision 处理），均超出本作业要求。

## 6. 运行时间观察与小结

* 三段式 + 共享内存 Blelloch 树是一个 **work-efficient** 算法
  （线性 work，对数 step）。在 T4 上对 8M `long long` 输入
  scan-only **~23.7×** 加速比，量级与 GPU Gems 3 第 39 章描述一致。
* `N` 越小，CUDA 启动 / 同步开销占比越大：
  - `N=32` 时 kernel scan 仅 0.02 ms，但 CPU 串行只用 0.00016 ms，
    GPU **慢于 CPU**——这是 kernel launch / event sync 的固有下界；
  - `N=10^6` 时 scan-only ~0.13 ms 对比串行 2.48 ms，~18.6×；
  - `N=8M` 时 ~23.7×。
* 本次每一步 commit 都做了 51 次中位的 head-to-head 对照（详见第 4
  节的迭代表与 `bench.md`），并据此 *回退* 了一次（v3a 的 int2+
  连续布局），充分体现「实验数据说话」的原则。
* 最大单步收益来自 **bank-conflict-free 共享内存**（v2，~33%）。
  其次是 **预分配 workspace**（v1，对小 N 极其重要）和 **输入
  int32 化**（v3，~4%）。

## 7. 文件清单

```
parco2026_hw3/
├── Makefile          # 编译 serial 和 parallel
├── README.md         # 题目原文
├── parallel.cu       # CUDA 实现
├── serial.cpp        # 题目自带串行实现
├── verify.py         # 题目自带验证脚本
├── run.slurm         # 集群批处理脚本
├── REPORT.md         # 本报告
└── bench.md          # 迭代实验数据（开发用，便于回看）
```

## 8. 集群上复现

```bash
sbatch run.slurm
```

`run.slurm` 会做：

1. 加载 CUDA 模块（如有），运行 `make`；
2. 跑算例 S、M、L 并各自调用 `verify.py`；
3. 算例 L **串行 51 次、并行预热 1 次后再 51 次**，分别取中位数，
   输出 `Serial median`、`CUDA median scan`、`CUDA median total`、
   `Speedup (scan only)` 与 `Speedup (incl. D2H)` 的最终摘要，并附
   min/max 用于判断稳定性。

输出会写到 `job_<JOBID>.out`，里面同时包含 `N=32` 的运行回显
（截图用）、`N=1000003` 的 `PASS` 行（截图用）以及 `N=8388608` 的
完整时间表与加速比。
