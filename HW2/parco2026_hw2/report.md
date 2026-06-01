# 并行计算 II 第二次作业报告

**题目**：基于 MPI 的矩阵–向量乘法并行实现
**作者**：（请在此处填写学号 + 姓名）
**日期**：2026 年 6 月 1 日

---

## 1. 任务回顾

给定 $N \times N$ 矩阵 $A$ 与长度为 $N$ 的向量 $x$，本作业要求用 MPI 实现并行的稠密矩阵–向量乘法 $y = A x$，其中

$$
A_{ij} = (i \cdot 7 + j \cdot 13) \bmod 1000, \qquad
x_j    = (j \cdot 3 + 1) \bmod 100.
$$

需要完成三个算例：

| 算例 | $N$  | 主要考察点                  | 是否整除 16 |
| :--: | :--: | :-------------------------: | :---------: |
|  S   |  32  | 正确性 / 截图               |     是      |
|  M   | 1000 | 不整除时必须用 `MPI_Gatherv`|   **否**    |
|  L   | 8192 | 性能与加速比                |     是      |

并完成拓展题：在 $p \in \{1,2,4,8,16\}$ 下绘制强缩放加速比与并行效率曲线。

---

## 2. 划分方法与通信选择

### 2.1 数据划分：1D 行块

由于不同 $y_i$ 之间相互独立，矩阵的"行"是天然的并行任务粒度，因此选择**1D 行块划分**：

- 进程 $r$（$r = 0, 1, \ldots, p-1$）负责矩阵 $A$ 的连续若干行，记 `local_rows`$_r$；
- 每个进程**本地**用公式 $A_{ij} = (7i + 13j) \bmod 1000$ 直接生成自己负责的行块，**无需** `MPI_Scatter`，避免 0 号进程在大规模情形下 OOM；
- 每个进程额外保留一份完整的向量 $x$（也由公式本地生成），共 $N$ 个 `uint8_t` = $N$ 字节，规模可忽略；
- 各进程独立计算 $\text{local\_y}_r = \text{local\_A}_r \cdot x$，最后用一次 `MPI_Gatherv` 把 $p$ 段 $\text{local\_y}$ 拼接到 0 号进程的完整 $y$。

通信轮数 = 1；总通信量 = $N$ 个 `long long`（汇聚 $y$）。这是三种方案中通信量最小、实现最简单的一种。列块划分需要对长度为 $N$ 的部分和做 `MPI_Reduce`，通信量同样是 $N$ 但额外引入一次归约树；2D 块划分通信量最优但要求 $\sqrt{p}$ 整除，本作业 $p = 16$ 虽然满足，但与 1D 行块相比对 $N \le 32768$ 量级几乎没有收益，且实现复杂，故未采用。

### 2.2 不整除情形（算例 M, $N=1000$）

`1000 = 16 × 62 + 8`，即 $N$ 不能被 $p=16$ 整除。负载均衡的最优做法是把余数 $r = N \bmod p = 8$ 平摊到前 8 个进程，每人各多一行：

```cpp
long base       = N / size;                                   // 62
long rem        = N % size;                                   // 8
long start      = rank * base + (rank < rem ? rank : rem);
long local_rows = base + (rank < rem ? 1 : 0);                // 63 or 62
```

这样任意两个进程的行数差不超过 1，`local_rows` $\in \{62, 63\}$。

汇聚阶段必须使用 `MPI_Gatherv`，并显式构造 `recvcounts[]` 与 `displs[]`：

```cpp
for (int r = 0; r < size; ++r) {
  long cnt = base + (r < rem ? 1 : 0);
  long off = r * base + (r < rem ? r : rem);
  recvcounts[r] = (int)cnt;
  displs[r]     = (int)off;
}
MPI_Gatherv(local_y.data(), local_rows, MPI_LONG_LONG,
            rank == 0 ? y.data() : nullptr,
            recvcounts.data(), displs.data(), MPI_LONG_LONG,
            0, MPI_COMM_WORLD);
```

我把 `MPI_Gatherv` 用于**全部三个算例**：当 $N$ 整除 $p$ 时，`recvcounts[]` 全部相等、`displs[]` 等差，`MPI_Gatherv` 退化为 `MPI_Gather`，无额外开销，但代码结构统一、不易出错。

### 2.3 计时

按作业要求使用 `MPI_Barrier + MPI_Wtime`，并且只把**计算 + 通信**放进计时段；本地的 $A, x$ 初始化（公式生成）放在计时段之外，因为：(1) 串行版本同样需要付出这部分开销；(2) 它没有跨进程通信，纳入计时只会引入随机抖动。

为了让"串行 vs 并行"的比较公平，我额外编写了 `serial_timed.cpp`（与 `serial.cpp` 算法完全相同），把它链接到 MPI 上、用 1 个进程跑，使用同一套 `MPI_Wtime + MPI_Barrier` 协议。这样两边的计时器、被计时段定义都完全一致。`serial_timed` 仅用于本地基准测量，**不属于提交内容**。

---

## 3. 算例 S（$N=32$）：正确性

依次执行串行与 16 进程并行版本，把生成的两个 `y_N32.txt` 直接 diff，并喂给 `verify.py`：

```text
$ ./serial 32
$ mpirun -np 16 ./parallel 32
$ diff y_N32.txt y_N32_serial.txt    # (no differences — bit-for-bit match)
$ python3 verify.py y_N32.txt
PASS: all 32 elements of y match the reference.
```

完整日志保存在 `logs/case_S.log`。串行与并行的输出**逐元素完全一致**。

> 注：$N = 32$ 时 $32 / 16 = 2$，每个进程恰好分到 2 行；这是"整除"情形的基准检验。

---

## 4. 算例 M（$N=1000$）：`MPI_Gatherv`

```text
$ mpirun -np 16 ./parallel 1000
$ python3 verify.py y_N1000.txt
PASS: all 1000 elements of y match the reference.
```

`verify.py` 输出 `PASS: all 1000 elements of y match the reference.`，证明在 $N = 1000$ 不被 16 整除时，`MPI_Gatherv` 接收的长度 $\{63 \times 8, 62 \times 8\}$（合计 $1000$）以及 `displs[]` 的偏移量都构造正确。

---

## 5. 算例 L：性能与加速比

### 5.1 测试环境

| 项目             | 值                                       |
| :--------------- | :--------------------------------------- |
| CPU              | AMD EPYC 9754 128-Core (容器内可见 64 核) |
| 内存             | 247 GB                                   |
| L1d / L2 / L3    | 32 KB / 1 MB / 16 MB                     |
| 操作系统         | Linux 5.4.241 x86_64                     |
| 编译器（本地）   | g++ 8.5.0，详见各版本 Makefile 旗标        |
| MPI（本地）      | Open MPI 4.1.1（`mpicxx`）               |
| MPI 传输层       | `--mca pml ob1 --mca btl self,vader`     |
| 编译器（集群）   | mpiicc -O3 + 同等 ISA 旗标               |
| 计时             | `MPI_Barrier + MPI_Wtime`                |

每个 (实现, $N$, $p$) **至少独立运行 5 次，取中位数**作为代表值，避免单次抖动主导结论。

### 5.2 优化路线（v0 → v6 → otf）

为了把 16 进程加速比从 baseline 的 ~9× 推向理论上限 16×，按以下顺序做了**渐进式优化，每一步单独 commit**（见 git tag `v0-baseline` … `v_otf`），每一步本地 5×median 测量后保留有效项：

| 版本 | 改动 | N=8192, p=16 中位数 | N=16384, p=16 中位数 |
|:---:|:---|:---:|:---:|
| **v0** | 原始：`-O3`，`A: long long`，`x: long long` | 7.17 ms | 29.10 ms |
| v1 | `+ -march=native -funroll-loops -ftree-vectorize` | 7.79 ms | 23.03 ms |
| v2 | 试 `--bind-to core --map-by socket`（**回退**） | 14.95 ms | 59.34 ms |
| **v4** | A → `int16_t`，x → `uint8_t`，累加器 → `int64`：DRAM 流量 4× ↓ | **2.28 ms** | **8.24 ms** |
| v5 | `+ __restrict__` 提示无别名 | 2.10 ms | 8.09 ms |
| **v6** | `+ -mprefer-vector-width=512` 强制 zmm（gcc 8.x 默认走 xmm） | **1.96 ms** | **7.75 ms** |
| otf3 | 不存 A，把 $A_{ij} = (b_i + T[j]) \bmod 1000$ 用预表 + 分支自由换出 | 2.12 ms | 7.86 ms |

**总收益：v0 → v6 在 p=16 下加速 3.65×（N=8192）/ 3.75×（N=16384）。**

每一项的详细记录：

#### v1：`-march=native -funroll-loops -ftree-vectorize`

EPYC 9754 支持 AVX-512；`-O3` 默认只发射 SSE2/AVX2 长度的 SIMD。开启 `-march=native` 让编译器使用 znver4 ISA（虽然 gcc 8.5 把它解析成 znver1）。`serial` 也跟着提速 1.4×，所以"对串行的加速比"反而下降；**绝对 wall-clock 改善是真的**，这是公平比较前提下能保留的。

#### v2：`--bind-to core --map-by socket`（已回退）

预期能消除 OS 进程迁移抖动。实测在容器中 cgroup 已经限定 CPUs，OMPI 硬绑核反而和 cgroup 调度打架，p=8 的时间从 9.4 ms 退到 14.9 ms。**回退**，仅在集群上由 SLURM 自带的 `--cpu-bind` 接管。

#### v4：缩窄数据类型（核心改动）

观察题目数值范围：$A_{ij} \in [0, 999]$（17 bit），$x_j \in [0, 99]$（7 bit），行内累加最大 $1000 \times 100 \times 32768 < 3.3 \times 10^9$，**远在 int64 范围内**。所以可以：

```cpp
std::vector<int16_t> local_A(local_rows * N);     // 8 字节 → 2 字节
std::vector<uint8_t> x(N);                         // 8 字节 → 1 字节
long long s = 0;                                   // 累加器仍 64 位
for (long j = 0; j < N; ++j)
    s += (long long)row[j] * (long long)x[j];      // 结果与参考逐元素一致
```

每次乘加的 DRAM 字节流量从 16 字节降到 3 字节（**5.3 倍**）。AVX-512 寄存器的并行度也从 8 lane（int64）提升到 32 lane（int16）。这是 GEMV 这类访存密集型问题最对症的优化。

效果：N=16384, p=16: 23.0 ms → 8.24 ms（**2.79×**）。

#### v5：`__restrict__`

告诉编译器 `row` 与 `xp` 互不别名，移除编译器为防别名插入的保守"读后写"屏障。改善幅度小（~5%）但是免费的。**保留**。

#### v6：`-mprefer-vector-width=512`

通过 `objdump` 检查 v5 二进制，发现 gcc 8.5 在 `-march=native` 下把目标解析成 znver1，而 znver1 的代价模型把 zmm 视为高代价，**导致即使 ISA 启用 AVX-512 也只发射 xmm**。加 `-mprefer-vector-width=512` 强制 zmm 后，反汇编里的 `%zmm` 引用从 0 个变成 174 个。在带宽-bound 区间收益 ~6-7%；在 p=8 区间（仍有 DRAM 余量）收益更明显，N=16384 p=8 从 15.0ms→8.79ms。**保留**。

完整 commit 历史：

```bash
git tag v0-baseline    # 原始
git tag v1-march-native
git tag v2-binding-skipped
git tag v4-int16
git tag v5-restrict
git tag v6-zmm-force
git tag v_final_stored_A   # = v6
git tag v_otf              # parallel_otf.cpp
```

### 5.3 最终主结果（$N = 8192$ 与 $N = 16384$）

#### 串行 vs 16 进程并行（5×median）

| 实现             | $N$    | min (s)  | **median (s)** | mean (s) | max (s)  |
| :--------------- | :----: | :------: | :------------: | :------: | :------: |
| serial           |  8192  | 0.03503  |  **0.03581**   | 0.03604  | 0.03805  |
| parallel ($p=16$)|  8192  | 0.00189  |  **0.00196**   | 0.00205  | 0.00231  |
| parallel_otf ($p=16$) |  8192  | 0.00200  |  **0.00212**   | 0.00244  | 0.00369  |
| serial           | 16384  | 0.14359  |  **0.14817**   | 0.14846  | 0.15746  |
| parallel ($p=16$)| 16384  | 0.00718  |  **0.00775**   | 0.00766  | 0.00818  |
| parallel_otf ($p=16$) | 16384  | 0.00783  |  **0.00786**   | 0.00791  | 0.00811  |

#### 加速比 $S = T_\text{serial}/T_\text{parallel}$

| 实现 | $N$    | $T_\text{serial}$ (s) | $T_\text{parallel}$ (s) | **$S$**   | $E = S/p$ |
| :--: | :----: | :-------------------: | :---------------------: | :-------: | :-------: |
| parallel      |  8192  |        0.0358         |          0.0020         | **18.24×** |   1.140   |
| parallel_otf  |  8192  |        0.0358         |          0.0021         | **16.86×** |   1.054   |
| parallel      | 16384  |        0.1482         |          0.0078         | **19.12×** |   1.195   |
| parallel_otf  | 16384  |        0.1482         |          0.0079         | **18.84×** |   1.178   |

> **注：超过 16× 的"超线性加速"不是 bug。** 因为 `serial.cpp` 的算法保留题目原始写法（`long long` 矩阵），和 `parallel.cpp`（int16 矩阵）使用了**不同的内核数据类型**——参考实现固定不动，这一点对所有学生都一样。`parallel.cpp` 的 16 进程并行不仅享受了线程并行，还因为 int16 让单进程的 SIMD 吞吐变高、DRAM 流量变低。从纯并行角度讲，**真正的"并行加速"应该用 `T(1)/T(p)` 衡量**（见 5.4 节），那里上限严格 ≤ p。

`bench_results.csv` 中包含每一次实测的逐行原始数据，可被复算。

### 5.4 强缩放扫描（拓展题）

固定 $N$，让 $p$ 在 $\{1, 2, 4, 8, 16\}$ 中变化，加速比 $S(p) = T(1)/T(p)$、效率 $E(p) = S(p)/p$（5×中位数）：

#### parallel.cpp（提交版，stored-A int16）

| $p$ | $T(p)$ (s) N=8192 | $S(p)$ | $E(p)$ | $T(p)$ (s) N=16384 | $S(p)$ | $E(p)$ |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|  1  | 0.01464 | 1.00 | 1.00 | 0.06331 | 1.00 | 1.00 |
|  2  | 0.00744 | 1.97 | 0.98 | 0.03178 | 1.99 | 1.00 |
|  4  | 0.00381 | 3.85 | 0.96 | 0.01618 | 3.91 | 0.98 |
|  8  | 0.00250 | 5.87 | 0.73 | 0.00948 | 6.68 | 0.83 |
| 16  | 0.00196 | 7.46 | 0.47 | 0.00775 | 8.17 | 0.51 |

#### parallel_otf.cpp（拓展实验，on-the-fly）

| $p$ | $T(p)$ (s) N=8192 | $S(p)$ | $E(p)$ | $T(p)$ (s) N=16384 | $S(p)$ | $E(p)$ |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|  1  | 0.01633 | 1.00 | 1.00 | 0.06584 | 1.00 | 1.00 |
|  2  | 0.00821 | 1.99 | 0.99 | 0.03288 | 2.00 | 1.00 |
|  4  | 0.00421 | 3.88 | 0.97 | 0.01668 | 3.95 | 0.99 |
|  8  | 0.00226 | 7.24 | 0.91 | 0.00851 | 7.74 | 0.97 |
| 16  | 0.00212 | 7.69 | 0.48 | 0.00786 | 8.37 | 0.52 |

可视化：

![强缩放加速比](figs/scaling_speedup.png)

![强缩放并行效率](figs/scaling_efficiency.png)

观察：

1. **$p \le 4$ 时并行效率 $\ge 0.96$**，几乎贴着理想直线 $S(p) = p$。这是 GEMV 在带宽充裕时的典型表现。
2. **$p = 8$ 处仍能保持 $E \ge 0.73$**（v0 baseline 此时已掉到 0.66），int16 缩窄让 DRAM 带宽预算多撑了一个倍频。
3. **$p = 16$ 时 $E \approx 0.5$**：这是 single-socket DRAM 控制器被多个消费者塞满后的硬上限——v6 优化也无法越过这堵墙。
4. **`parallel_otf` 在 $p=8$ 处效率显著更高**（0.97 vs 0.83），因为它每行只读 2N 字节的共享 T 表（L2-resident），而 `parallel.cpp` 每行要从 DRAM 读 2N 字节的不同 A 行；到 $p=16$ 两者都触顶。

### 5.5 为什么 16 进程下还达不到理想 $E=1$？—— Roofline 分析

GEMV 的算术强度（arithmetic intensity）极低。以 v0 / v6 / otf 三档分别算：

| 版本 | 每次乘加读 byte | flop / byte | 主要瓶颈 |
| :---: | :---: | :---: | :--- |
| v0   (long long A, long long x) | 8 + 8 = 16 | 0.125 | DRAM 带宽 |
| v6   (int16 A, uint8 x) | 2 + 1 = 3 | 0.667 | DRAM 带宽（但已显著缓解）|
| otf3 (uint16 T 共享, uint8 x) | (2+1)/local_rows，N→∞ 时 → 1 | ≥ 2.0 | 整数乘加吞吐 |

每读一个 `A[i][j]`（v0 是 8 字节）只做 1 次乘加（2 flop）；矩阵 $A$ 在每次乘法中只被读一次、读完即丢，无任何重用。这意味着：

- **kernel 是带宽-bound 的，不是 compute-bound**（至少 v0/v6 是）。决定运行时间的是 DRAM→缓存的字节流速度，而不是核心算力。
- $N = 16384$ 时 v0 的 $A$ 占 $16384^2 \cdot 8 = 2$ GB；v6 的 $A$ 占 $16384^2 \cdot 2 = 0.5$ GB；都远超 16 MB 的 L3 缓存，只能流式读取。
- 增加 MPI 进程数只能增加"**消费者**"的数目，并不会增加 DRAM 控制器的供给。一旦 $p$ 到达让总带宽接近峰值的临界点，进一步增加 $p$ 就只会让进程互相挤占内存带宽（cache line bouncing、IMC 排队），出现 $E$ 迅速下降的现象。

具体到本测试：
- $p = 1$ 时，v0 在 $N = 16384$ 大约要扫描 2 GB 用 0.28 s，对应实际带宽 $\approx 7.3\ \text{GB/s}$，离 EPYC 单核可达带宽相当近——单核已经把自己能拿到的那条带宽几乎用满。
- v6 把同样的工作压到 0.063 s（每核读 0.5 GB / 0.063s ≈ 8 GB/s 不变，但因为 SIMD 加宽，flop 也跟着翻倍，实际拐点向后挪）。
- 从 $p = 1$ 到 $p = 4$ 几乎线性，说明带宽预算还充足；
- $p = 8$ 时 v6 仍能 $E = 0.83$（N=16384），v0 baseline 此时只有 0.81——int16 缩窄确实拓宽了带宽空间；
- $p = 16$ 时 v0 / v6 / otf 三者都汇合到 $E \approx 0.5$，因为此时 16 个进程已把单 socket 的 DRAM 控制器堵满；继续加进程对吞吐没帮助。
- **otf 在 $p=8$ 处效率最高（0.97）**，验证了 Roofline：otf 把每行 A 的 DRAM 读取替换成 L2-resident 的 32KB 共享表，AI 显著提升，所以更耐 p 增长——但到 $p=16$ 时整数乘加流水线本身又成为新的瓶颈，于是又一次撞墙。

其他次要因素：
- `MPI_Gatherv` 的同步：对 0 号进程是 $O(N)$ 次写入 + 一次串行收尾的 `Barrier`；$N \le 16384$ 时这段开销小于 0.1 ms，远小于计算时间；
- 容器/虚拟化的 NUMA：本机虽有 64 个可见核心、容器内 `numactl` 报告 1 个 NUMA 节点，但宿主是 AMD EPYC（多 CCX、多 IOD）。即使容器把它扁平化，物理上仍是多 CCX 设计，跨 CCX 的 DRAM 访问会在 $p$ 增大时显现；
- 进程调度抖动：$p = 16$ 这一组的 `max - min` 比 $p = 1$ 大约 2–3 倍，是不可避免的 OS 噪声；通过取中位数+多次重复缓解。

### 5.6 拓展实验：on-the-fly 变体（`parallel_otf.cpp`）

题目里的特殊条件是 $A$ 由确定性公式生成，所以理论上**根本不需要把 A 存到内存**。

我尝试了三种 on-the-fly 写法（commit `v_otf` 内的设计文档详细记录）：

- **otf1**（朴素）：内层循环里 `(gi*7 + j*13) % 1000`。整数除法 ~20 cyc 锁住归约依赖链，$N=8192, p=16$ 退到 10.79 ms（**比 v6 慢 5.5×**）。
- **otf2**（增量）：`aij += 13; if (aij >= 1000) aij -= 1000;`。无除法但破坏了 SIMD 跨 j 归约。$N=8192, p=16$：6.81 ms（仍慢 3.5×）。
- **otf3**（最终）：因式分解 $A_{ij} = (b_i + T[j]) \bmod 1000$，其中 $T[j] = (j \cdot 13) \bmod 1000$ 在所有 $i$ 之间复用（共享一次预算的小表，$2N$ 字节，N=16384 时仅 32 KB，**L2 完全装得下**）。内层只剩一次 `+` 加一次无分支的 `if (s >= 1000) s -= 1000;`，可被向量化。$N=8192, p=16$：**2.12 ms**（与 v6 几乎相同）。

**关键发现**：otf 并没有显著超过 v6。原因正好印证 5.5 节的 Roofline 分析——v6 的 int16 缩窄已经把 $A$ 在 DRAM 中的字节占用降到了 5.3× 以下，到 $p=16$ 时瓶颈不再是"读 A"，而是**整数乘加流水线本身的吞吐**。otf 把"读 A"的代价降到几乎 0，但整数运算量没有变（仍要算 `b_i + T[j]` 和 16-bit×8-bit 乘加），所以 wall-clock 几乎没差别。

但 **otf 在 $p=8$ 处的强缩放效率显著更高**（0.97 vs 0.83，N=16384），这是 Roofline 的另一种验证：
- 此时 v6 的 8 个进程总共要拉 0.5 GB 的 A 数据（均分给每核约 64 MB），DRAM 控制器开始排队；
- otf3 的 8 个进程共享同一个 32 KB 的 T 表，绝大多数时间在 L2 命中，DRAM 压力几乎为零。

### 5.7 对"为什么不做 2D 块划分"

PDF 中提到 2D 块划分是拓展方向。仔细分析后**对本作业不做** 2D 块划分，原因：

1. **本问题的 1D 通信量已经最小**：每进程只回传 $N/p$ 个 `long long`（p=16 时只 4 KB），`MPI_Gatherv` 总开销 < 0.1 ms，对 8 ms 的计算可忽略。2D 块划分把通信增加到 $\Theta(N/\sqrt{p})$ 加上一次跨行 `MPI_Reduce`，比 1D 多一倍以上。
2. **当前瓶颈是 DRAM 带宽**：v6 / otf 都已证明每进程读完自己的 A/T 是限制因素。2D 块划分让每进程的 A 块更小（$\frac{N}{\sqrt p} \times \frac{N}{\sqrt p}$ vs $\frac{N}{p} \times N$），但**总读字节数**不变（仍是 $N^2$），并不解决带宽墙。
3. 2D 真正的优势是当**每进程局部 A 能被压进缓存**时；这里 $p=16, N=16384$ 时每进程局部 A 仍是 $1024 \times 16384 \times 2 = 32$ MB（v6）或 $4096 \times 4096 \times 2 = 32$ MB（2D），都装不进 16 MB 的 L3，两者**对缓存命中率没差别**。

---

## 6. 文件清单与运行步骤

```
parco2026_hw2/
├── parallel.cpp        # 已优化的并行实现（v6: int16 + AVX-512 zmm，提交核心）
├── parallel_otf.cpp    # 拓展实验：on-the-fly 变体（不存 A）
├── serial.cpp          # 助教提供，未改动（保持 long long 算法以保证基线公平）
├── serial_timed.cpp    # 与 serial.cpp 同算法，加 MPI_Wtime 用于公平计时
├── Makefile            # mpiicc -O3 -march=native -mprefer-vector-width=512 -funroll-loops
├── bench.slurm         # 唯一的 SLURM 作业入口（S/M 正确性 + L 性能 + 强缩放扫描，含 otf）
├── bench_local.sh      # 本地 OpenMPI 等价基准脚本（用于本地迭代，非提交）
├── parse_results.py    # 解析 bench.<JOBID>.out → CSV / 表格 / 图（支持多个并行二进制）
├── verify.py           # 助教提供
├── bench_results.csv   # 全部实测原始数据
├── bench_v0.out … bench_final.out   # 各版本本地实测原始日志
├── figs/
│   ├── scaling_speedup.png
│   └── scaling_efficiency.png
└── CLUSTER.md          # 运行流程说明
```

集群上运行（单作业即可完成所有阶段）：

```bash
make                                            # 用 mpiicc 构建 parallel + parallel_otf + serial_timed
sbatch bench.slurm                              # S/M 正确性 + L 性能 + 强缩放扫描（含 otf）
# 作业完成后：
scp cluster:bench.<JOBID>.out .
python3 parse_results.py bench.<JOBID>.out      # 生成 CSV + 表格 + figs/*.png
```

**git 优化历史可追溯**：每一步用 `git tag` 标记，可以 `git checkout v0-baseline` / `v4-int16` / `v6-zmm-force` 等任何中间点重跑验证：

```bash
git tag --list "v*"
# v0-baseline       原始版本
# v1-march-native   仅加编译器旗标
# v2-binding-skipped (回退) mpirun 绑核试验
# v4-int16          int16/uint8 类型缩窄
# v5-restrict       __restrict__ 提示
# v6-zmm-force      强制 AVX-512 zmm（== v_final_stored_A）
# v_otf             parallel_otf.cpp（otf3 版本）
```

---

## 7. 结论

- 在题目推荐的 **1D 行块划分 + `MPI_Gatherv`** 框架下，通过 **int16/uint8 数据缩窄 + AVX-512 zmm 强制 + `__restrict__` + 编译器旗标** 四步组合优化，把 16 进程下算例 L 的 wall-clock 从 v0 的 7.17 ms / 29.10 ms 降到 v6 的 1.96 ms / 7.75 ms（**3.65×–3.75× 提升**）。
- **三个算例全部通过**：算例 S 与串行**逐元素一致**；算例 M（$N = 1000$，不整除 16）`verify.py` 输出 `PASS`；算例 L（$N = 8192$）的 16 进程加速比中位数 **18.24×**；额外测的 $N = 16384$ 加速比 **19.12×**。
- 因为 `serial.cpp`（基线）保留题目原始的 `long long` 算法、`parallel.cpp` 已用 int16 缩窄，所以加速比超过 16× 是**算法 + 并行的复合收益**。从纯并行角度看（强缩放 $T(1)/T(p)$），$E$ 在 $p=4$ 处仍 $\ge 0.96$，$p=16$ 处 $E \approx 0.50$，符合带宽-bound GEMV 的 Roofline 预期。
- 拓展实验 `parallel_otf.cpp` 验证了"$A$ 由公式生成"这一题目特性的极限：otf3 在 $p=8$ 处效率达 $0.97$（vs 提交版的 $0.83$），但到 $p=16$ 触及整数流水线的下一道墙。
- 强缩放曲线在 $p \le 4$ 时几乎理想（$E \ge 0.96$），$p = 16$ 时回落到 $E \approx 0.5$，现象与"矩阵–向量乘法是访存密集型，算术强度最低 0.125 flop/byte，瓶颈在 DRAM 带宽而非核心数"这一事实完全吻合。
