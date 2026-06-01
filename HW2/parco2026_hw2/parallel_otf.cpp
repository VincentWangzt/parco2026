/*
 * Foundations of Parallel Computing II.
 * MPI parallel implementation of matrix-vector multiplication y = A * x.
 *
 *   --- ON-THE-FLY VARIANT (extra-credit experiment) ---
 *
 * Identical to parallel.cpp except: A is *never* materialised. Each
 * inner-loop iteration recomputes A_ij = (i*7 + j*13) % 1000 in registers
 * and immediately consumes it. The motivation is that A is defined by a
 * deterministic closed-form formula, so storing it in DRAM is pure waste —
 * the kernel can be made compute-bound instead of memory-bound.
 *
 * Memory traffic per multiply-add (row-major, large N):
 *   parallel.cpp (stored A, int16):   2 bytes A + 1 byte x   = 3 bytes
 *   parallel_otf.cpp (on-the-fly A):  0 bytes A + 1 byte x   = 1 byte
 * Plus a one-time read-once for x and write-once for y, both negligible.
 *
 * Roofline:
 *   AI(otf) = 2 flop / 1 byte = 2 flop/byte
 *   With single-rank DRAM bandwidth ~7 GB/s, the bandwidth ceiling alone
 *   would be 14 Gflops; in practice once AI > the machine balance point
 *   the kernel becomes compute-bound and the bottleneck becomes the
 *   integer multiply-add throughput, where 16 ranks scale near-linearly.
 *
 * Why is this an extra-credit experiment, not the submitted version?
 *   The assignment is "matrix-vector multiplication"; recomputing A is
 *   only legal because the assignment also specifies that A is generated
 *   by a formula on each process. In a real GEMV (A from disk, network,
 *   or the previous kernel's output), this trick is impossible.
 *   parallel.cpp keeps the standard "store A, multiply by x" kernel.
 *
 * Optimisations on top of v6:
 *   - inner-loop fused: A_ij = (base_i + T[j]) mod 1000
 *   - T[j] = (j*13) mod 1000 precomputed ONCE per process (2N bytes,
 *     L2-resident at N=16384, total bandwidth ~32 KB instead of 16 MB)
 *   - branchless wraparound: `s = base_i + T[j]; if (s >= 1000) s -= 1000;`
 *
 * Note that the timed section excludes the cost of building T (just like
 * parallel.cpp's timed section excludes the cost of building local_A).
 *
 * Usage: mpirun -np <P> ./parallel_otf <N>
 */

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

inline uint8_t x_elem(long j) {
  return static_cast<uint8_t>((j * 3LL + 1) % 100);
}

int main(int argc, char* argv[]) {
  long N;
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 2)
    N = 32;
  else
    N = std::atol(argv[1]);

  if (rank == 0) {
    std::cout << "Executing on-the-fly with N = " << N
              << ", comm size = " << size << "\n";
  }

  // 1D row-block partitioning, identical to parallel.cpp
  long base       = N / size;
  long rem        = N % size;
  long start      = rank * base + (rank < rem ? rank : rem);
  long local_rows = base + (rank < rem ? 1 : 0);

  // Each process keeps a full copy of x (uint8); ~N bytes is trivial.
  std::vector<uint8_t> x(N);
  for (long j = 0; j < N; ++j) x[j] = x_elem(j);

  // Precompute T[j] = (j*13) mod 1000 once per process. 2*N bytes as
  // uint16 — at N=16384 that's 32 KB, fits in L2. Inside the timed loop
  // we read T sequentially and combine with a per-row scalar `base_i`,
  // turning A row generation into a 1-byte-per-element streaming load
  // from a small reusable buffer instead of from a 256 MB A array.
  // T is the SAME for every row, so its DRAM bandwidth cost is paid once.
  // Excluded from the timed section (parallel.cpp's local_A init is also
  // excluded, so this matches the assignment's "init outside timing" rule).
  std::vector<uint16_t> Tbuf(N);
  for (long j = 0; j < N; ++j)
    Tbuf[j] = static_cast<uint16_t>((j * 13LL) % 1000);

  // recvcounts/displs for MPI_Gatherv (works for divisible & non-divisible N).
  std::vector<int> recvcounts(size), displs(size);
  for (int r = 0; r < size; ++r) {
    long cnt = base + (r < rem ? 1 : 0);
    long off = r * base + (r < rem ? r : rem);
    recvcounts[r] = static_cast<int>(cnt);
    displs[r]     = static_cast<int>(off);
  }

  std::vector<long long> y;
  if (rank == 0) y.resize(N);

  std::vector<long long> local_y(local_rows, 0);

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  //   otf3 (kept): factor A_ij = (base_i + T[j]) mod 1000, where
  //                T[j] = (j*13) mod 1000 is precomputed ONCE per process
  //                (size 2*N bytes as int16, ~32 KB at N=16384, L2-resident).
  //         The inner loop becomes a SIMD-friendly read from T plus a
  //         branchless wrap-around (`s = base_i + T[j]; if (s >= 1000) s -= 1000;`)
  //         — the compiler turns it into a vpminuw or a cmov on AVX-512.
  //         Memory traffic: 2*N bytes total for all local_rows iterations
  //         (T is reused), versus 2*N*local_rows in parallel.cpp.
  //         At N=16384, p=16: parallel.cpp reads 16 MB of A; otf3 reads
  //         32 KB once. That's the entire reason this variant exists.
  const uint8_t*  __restrict__ xp = x.data();
  const uint16_t* __restrict__ Tp = Tbuf.data();
  for (long i = 0; i < local_rows; ++i) {
    const long gi = start + i;
    const uint16_t base_i = static_cast<uint16_t>((gi * 7LL) % 1000);
    long long s = 0;
    for (long j = 0; j < N; ++j) {
      // sum is in [0, 1998]; one branchless wrap brings it to [0, 999].
      uint32_t sum = static_cast<uint32_t>(base_i) + static_cast<uint32_t>(Tp[j]);
      if (sum >= 1000) sum -= 1000;
      s += static_cast<long long>(sum) * static_cast<long long>(xp[j]);
    }
    local_y[i] = s;
  }

  MPI_Gatherv(local_y.data(),
              static_cast<int>(local_rows),
              MPI_LONG_LONG,
              rank == 0 ? y.data() : nullptr,
              recvcounts.data(),
              displs.data(),
              MPI_LONG_LONG,
              0,
              MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();
  if (rank == 0) {
    std::cout << "time=" << (t1 - t0) << " s\n";
  }

  if (rank == 0) {
    std::string filename = "y_N" + std::to_string(N) + ".txt";
    std::ofstream ofs(filename);
    if (!ofs) {
      std::cerr << "ERROR: cannot open output file " << filename << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (long i = 0; i < N; ++i) ofs << y[i] << "\n";
    ofs.close();
    std::cout << "Wrote result to " << filename << "\n";

    if (N <= 100) {
      std::cout << "y =";
      for (long i = 0; i < N; ++i) std::cout << " " << y[i];
      std::cout << "\n";
    }
  }

  MPI_Finalize();
  return 0;
}
