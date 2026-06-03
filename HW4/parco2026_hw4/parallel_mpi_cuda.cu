/*
 * Foundations of Parallel Computing II.
 * MPI + CUDA hybrid implementation of Conway's Game of Life.
 *
 * Decomposition: 1-D row partition across MPI ranks. Each rank holds its
 * own (local_rows + 2) x N device buffer (rows 0 and local_rows+1 are
 * ghost rows). Each step the rank:
 *   1. Copies its top and bottom rows from device -> host pinned buffers.
 *   2. Exchanges them with neighbors via MPI_Sendrecv.
 *   3. Copies the received ghost rows host -> device.
 *   4. Launches a CUDA kernel that updates only the interior local_rows.
 *   5. Swaps the two device buffers.
 * After T steps each rank copies its rows back to the host and rank 0
 * gathers the full grid via MPI_Gatherv.
 *
 * On a node with a single GPU all ranks share that GPU (selected via
 * cudaSetDevice using rank % deviceCount). The decomposition still
 * demonstrates the MPI+CUDA halo-exchange pattern that scales when more
 * than one GPU is available.
 *
 * Usage: mpirun -np P ./parallel_mpi_cuda <N> <T>
 */

#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                   __LINE__, cudaGetErrorString(_err));                        \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0)

inline int init_cell(int i, int j) {
  return ((i * 131 + j * 17 + 7) % 100) < 30 ? 1 : 0;
}

// Kernel updates rows [1, local_rows] of a (local_rows + 2) x N grid.
__global__ void life_step_kernel(const unsigned char* __restrict__ in,
                                 unsigned char* __restrict__ out, int N,
                                 int local_rows) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;  // skip ghost row 0
  if (j >= N || i > local_rows) return;

  int cnt = 0;
#pragma unroll
  for (int di = -1; di <= 1; ++di) {
#pragma unroll
    for (int dj = -1; dj <= 1; ++dj) {
      if (di == 0 && dj == 0) continue;
      int nj = j + dj;
      if (nj >= 0 && nj < N) {
        cnt += in[(i + di) * N + nj];
      }
    }
  }
  int alive = in[i * N + j];
  out[i * N + j] =
      (unsigned char)((alive ? (cnt == 2 || cnt == 3) : (cnt == 3)) ? 1 : 0);
}

static void write_grid(const std::vector<unsigned char>& grid, int N, int T) {
  std::string filename = "life_N" + std::to_string(N) + "_T" + std::to_string(T) + ".txt";
  std::ofstream ofs(filename);
  if (!ofs) {
    std::cerr << "ERROR: cannot open output file " << filename << "\n";
    std::exit(1);
  }
  ofs << N << " " << N << "\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      ofs << static_cast<int>(grid[i * N + j]);
      if (j + 1 < N) ofs << " ";
    }
    ofs << "\n";
  }
  std::cout << "Wrote result to " << filename << "\n";
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = 16, T = 4;
  if (argc >= 2) N = std::atoi(argv[1]);
  if (argc >= 3) T = std::atoi(argv[2]);

  // Pick a GPU. With a single device all ranks share it (oversubscribe).
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    if (rank == 0) std::cerr << "No CUDA devices found.\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CUDA_CHECK(cudaSetDevice(rank % device_count));

  if (rank == 0) {
    std::cout << "Executing MPI+CUDA Game of Life with N = " << N
              << ", T = " << T << ", P = " << size
              << ", GPUs = " << device_count << "\n";
  }

  // Row partition.
  std::vector<int> counts(size), offsets(size);  // cells
  std::vector<int> row_counts(size), row_offsets(size);
  {
    int base = N / size;
    int rem = N % size;
    int off = 0;
    for (int r = 0; r < size; ++r) {
      int rows_r = base + (r < rem ? 1 : 0);
      row_counts[r] = rows_r;
      row_offsets[r] = off;
      counts[r] = rows_r * N;
      offsets[r] = off * N;
      off += rows_r;
    }
  }
  int local_rows = row_counts[rank];
  int row_start = row_offsets[rank];

  size_t stride = static_cast<size_t>(N);
  size_t local_h = static_cast<size_t>(local_rows + 2);

  // Initialize local rows on the host with the deterministic formula.
  std::vector<unsigned char> host_local(local_h * stride, 0);
  for (int i = 0; i < local_rows; ++i) {
    int gi = row_start + i;
    for (int j = 0; j < N; ++j) {
      host_local[(i + 1) * stride + j] =
          static_cast<unsigned char>(init_cell(gi, j));
    }
  }

  unsigned char *d_current = nullptr, *d_next = nullptr;
  CUDA_CHECK(cudaMalloc(&d_current, local_h * stride));
  CUDA_CHECK(cudaMalloc(&d_next, local_h * stride));
  CUDA_CHECK(cudaMemcpy(d_current, host_local.data(), local_h * stride,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_next, 0, local_h * stride));

  // Pinned host buffers for halo exchange (4 rows).
  unsigned char *h_send_top = nullptr, *h_send_bot = nullptr;
  unsigned char *h_recv_top = nullptr, *h_recv_bot = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_send_top, N));
  CUDA_CHECK(cudaMallocHost(&h_send_bot, N));
  CUDA_CHECK(cudaMallocHost(&h_recv_top, N));
  CUDA_CHECK(cudaMallocHost(&h_recv_bot, N));

  int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

  dim3 block(32, 8);
  dim3 grid((N + block.x - 1) / block.x,
            (local_rows + block.y - 1) / block.y);

  // Warm-up.
  life_step_kernel<<<grid, block>>>(d_current, d_next, N, local_rows);
  CUDA_CHECK(cudaDeviceSynchronize());
  // Reset state.
  CUDA_CHECK(cudaMemcpy(d_current, host_local.data(), local_h * stride,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_next, 0, local_h * stride));

  MPI_Barrier(MPI_COMM_WORLD);
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < T; ++step) {
    // 1. Copy boundary rows from device to pinned host buffers.
    CUDA_CHECK(cudaMemcpy(h_send_top, d_current + 1 * stride, N,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_send_bot, d_current + local_rows * stride, N,
                          cudaMemcpyDeviceToHost));

    // 2. Exchange halos.
    MPI_Sendrecv(h_send_top, N, MPI_UNSIGNED_CHAR, up, 0,
                 h_recv_bot, N, MPI_UNSIGNED_CHAR, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_send_bot, N, MPI_UNSIGNED_CHAR, down, 1,
                 h_recv_top, N, MPI_UNSIGNED_CHAR, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Fixed dead boundary at the global edges.
    if (up == MPI_PROC_NULL)   std::memset(h_recv_top, 0, N);
    if (down == MPI_PROC_NULL) std::memset(h_recv_bot, 0, N);

    // 3. Copy ghost rows back to device.
    CUDA_CHECK(cudaMemcpy(d_current + 0 * stride, h_recv_top, N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_current + (local_rows + 1) * stride, h_recv_bot,
                          N, cudaMemcpyHostToDevice));

    // 4. Launch the kernel; it updates rows 1..local_rows.
    life_step_kernel<<<grid, block>>>(d_current, d_next, N, local_rows);
    // 5. Swap.
    std::swap(d_current, d_next);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Copy the local rows back and gather on rank 0.
  CUDA_CHECK(cudaMemcpy(host_local.data() + stride, d_current + stride,
                        local_rows * stride, cudaMemcpyDeviceToHost));

  std::vector<unsigned char> full;
  if (rank == 0) full.resize(static_cast<size_t>(N) * N);
  MPI_Gatherv(host_local.data() + stride, local_rows * N, MPI_UNSIGNED_CHAR,
              rank == 0 ? full.data() : nullptr,
              counts.data(), offsets.data(), MPI_UNSIGNED_CHAR,
              0, MPI_COMM_WORLD);

  CUDA_CHECK(cudaFree(d_current));
  CUDA_CHECK(cudaFree(d_next));
  CUDA_CHECK(cudaFreeHost(h_send_top));
  CUDA_CHECK(cudaFreeHost(h_send_bot));
  CUDA_CHECK(cudaFreeHost(h_recv_top));
  CUDA_CHECK(cudaFreeHost(h_recv_bot));

  if (rank == 0) {
    std::cout << "MPI+CUDA compute time: " << ms << " ms\n";
    write_grid(full, N, T);
    if (N <= 32) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          std::cout << static_cast<int>(full[i * N + j])
                    << (j + 1 == N ? '\n' : ' ');
        }
      }
    }
  }

  MPI_Finalize();
  return 0;
}
