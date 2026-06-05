/*
 * Foundations of Parallel Computing II.
 * CUDA single-GPU implementation of Conway's Game of Life.
 *
 * One thread per cell. Two device buffers act as a double buffer; each step
 * reads from `d_current` and writes to `d_next`, then the host pointers are
 * swapped. The kernel pads neighbor lookups with the fixed dead boundary
 * (cells outside the N x N grid are zero).
 *
 * Initial grid: grid[i][j] = (((i*131 + j*17 + 7) % 100) < 30)
 *
 * Usage: ./parallel_cuda <N> <T>
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
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
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

inline int init_cell(int i, int j) {
  return ((i * 131 + j * 17 + 7) % 100) < 30 ? 1 : 0;
}

// Shared-memory tiling: each block of BX*BY threads loads a (BY+2)x(BX+2)
// tile (its cells plus a 1-cell halo) into __shared__ memory once, then every
// thread reads its 9 neighbours from shared memory rather than global memory.
// Halo cells outside the global grid are loaded as 0 (fixed dead boundary).
template <int BX, int BY>
__global__ void life_step_kernel(const unsigned char* __restrict__ in,
                                 unsigned char* __restrict__ out, int N) {
  __shared__ unsigned char tile[BY + 2][BX + 2];

  int j = blockIdx.x * BX + threadIdx.x;  // global column
  int i = blockIdx.y * BY + threadIdx.y;  // global row
  int tx = threadIdx.x + 1;               // tile-local column [1, BX]
  int ty = threadIdx.y + 1;               // tile-local row    [1, BY]

  auto load = [&](int gi, int gj) -> unsigned char {
    return (gi >= 0 && gi < N && gj >= 0 && gj < N) ? in[gi * N + gj]
                                                    : (unsigned char)0;
  };

  // Center cell.
  tile[ty][tx] = load(i, j);
  // Top / bottom halo rows.
  if (threadIdx.y == 0)        tile[0][tx]      = load(i - 1, j);
  if (threadIdx.y == BY - 1)   tile[BY + 1][tx] = load(i + 1, j);
  // Left / right halo cols.
  if (threadIdx.x == 0)        tile[ty][0]      = load(i, j - 1);
  if (threadIdx.x == BX - 1)   tile[ty][BX + 1] = load(i, j + 1);
  // 4 corners.
  if (threadIdx.x == 0      && threadIdx.y == 0)      tile[0][0]           = load(i - 1, j - 1);
  if (threadIdx.x == BX - 1 && threadIdx.y == 0)      tile[0][BX + 1]      = load(i - 1, j + 1);
  if (threadIdx.x == 0      && threadIdx.y == BY - 1) tile[BY + 1][0]      = load(i + 1, j - 1);
  if (threadIdx.x == BX - 1 && threadIdx.y == BY - 1) tile[BY + 1][BX + 1] = load(i + 1, j + 1);

  __syncthreads();

  if (i >= N || j >= N) return;

  int cnt = tile[ty - 1][tx - 1] + tile[ty - 1][tx] + tile[ty - 1][tx + 1]
          + tile[ty    ][tx - 1]                    + tile[ty    ][tx + 1]
          + tile[ty + 1][tx - 1] + tile[ty + 1][tx] + tile[ty + 1][tx + 1];
  int alive = tile[ty][tx];
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
  int N = 16, T = 4;
  if (argc >= 2) N = std::atoi(argv[1]);
  if (argc >= 3) T = std::atoi(argv[2]);

  std::cout << "Executing CUDA Game of Life with N = " << N << ", T = " << T
            << "\n";

  std::vector<unsigned char> host(static_cast<size_t>(N) * N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      host[i * N + j] = static_cast<unsigned char>(init_cell(i, j));

  size_t bytes = static_cast<size_t>(N) * N;
  unsigned char *d_current = nullptr, *d_next = nullptr;
  CUDA_CHECK(cudaMalloc(&d_current, bytes));
  CUDA_CHECK(cudaMalloc(&d_next, bytes));
  CUDA_CHECK(cudaMemcpy(d_current, host.data(), bytes, cudaMemcpyHostToDevice));

  constexpr int BX = 16, BY = 16;
  dim3 block(BX, BY);
  dim3 grid((N + BX - 1) / BX, (N + BY - 1) / BY);

  // Warm-up to surface init / JIT cost.
  life_step_kernel<BX, BY><<<grid, block>>>(d_current, d_next, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  // Reset state after warm-up: re-upload the initial grid.
  CUDA_CHECK(cudaMemcpy(d_current, host.data(), bytes, cudaMemcpyHostToDevice));

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int step = 0; step < T; ++step) {
    life_step_kernel<BX, BY><<<grid, block>>>(d_current, d_next, N);
    std::swap(d_current, d_next);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "CUDA compute time: " << ms << " ms\n";

  CUDA_CHECK(cudaMemcpy(host.data(), d_current, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_current));
  CUDA_CHECK(cudaFree(d_next));

  write_grid(host, N, T);

  if (N <= 32) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        std::cout << static_cast<int>(host[i * N + j])
                  << (j + 1 == N ? '\n' : ' ');
      }
    }
  }
  return 0;
}
