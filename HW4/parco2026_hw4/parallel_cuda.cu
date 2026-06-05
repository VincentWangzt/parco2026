/*
 * Foundations of Parallel Computing II.
 * CUDA single-GPU implementation of Conway's Game of Life.
 *
 * One thread per cell. Two device buffers act as a double buffer; each step
 * reads from `d_current` and writes to `d_next`, then the host pointers are
 * swapped.
 *
 * Halo-padded layout: device buffers are (N+2) x (N+2) with the outer ring
 * pinned to zero (fixed dead boundary). The kernel works on the interior
 * (1..N) x (1..N) and reads 8 neighbours unconditionally — no per-cell
 * bounds checks.
 *
 * Initial grid: grid[i][j] = (((i*131 + j*17 + 7) % 100) < 30)
 *
 * Usage: ./parallel_cuda <N> <T>
 */

#include <cuda_runtime.h>

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
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

inline int init_cell(int i, int j) {
  return ((i * 131 + j * 17 + 7) % 100) < 30 ? 1 : 0;
}

// Padded layout: one thread per interior cell. (i, j) in [1, N], and stride
// is N+2 so neighbours at (i±1, j±1) always land on real (possibly-zero) bytes.
__global__ void life_step_kernel(const unsigned char* __restrict__ in,
                                 unsigned char* __restrict__ out,
                                 int N, int stride) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > N || j > N) return;

  const unsigned char* up_r = in + (i - 1) * stride;
  const unsigned char* mid  = in + i       * stride;
  const unsigned char* dn_r = in + (i + 1) * stride;

  int cnt = up_r[j - 1] + up_r[j] + up_r[j + 1]
          + mid[j - 1]            + mid[j + 1]
          + dn_r[j - 1] + dn_r[j] + dn_r[j + 1];
  int alive = mid[j];
  out[i * stride + j] =
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

  // Build the initial grid into a padded (N+2) x (N+2) host buffer so the
  // outer ring (rows 0/N+1, cols 0/N+1) is zero by construction.
  int stride = N + 2;
  std::vector<unsigned char> host_padded(static_cast<size_t>(stride) * stride, 0);
  for (int i = 0; i < N; ++i) {
    unsigned char* row = &host_padded[(i + 1) * stride + 1];
    for (int j = 0; j < N; ++j) {
      row[j] = static_cast<unsigned char>(init_cell(i, j));
    }
  }

  size_t bytes = static_cast<size_t>(stride) * stride;
  unsigned char *d_current = nullptr, *d_next = nullptr;
  CUDA_CHECK(cudaMalloc(&d_current, bytes));
  CUDA_CHECK(cudaMalloc(&d_next, bytes));
  // Initialise BOTH buffers from the padded host buffer so the outer ring of
  // d_next stays zero across iterations (the kernel never writes to it).
  CUDA_CHECK(cudaMemcpy(d_current, host_padded.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_next,    host_padded.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  // Warm-up to surface init / JIT cost.
  life_step_kernel<<<grid, block>>>(d_current, d_next, N, stride);
  CUDA_CHECK(cudaDeviceSynchronize());
  // Reset state after warm-up: re-upload the initial grid into both buffers.
  CUDA_CHECK(cudaMemcpy(d_current, host_padded.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_next,    host_padded.data(), bytes, cudaMemcpyHostToDevice));

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int step = 0; step < T; ++step) {
    life_step_kernel<<<grid, block>>>(d_current, d_next, N, stride);
    std::swap(d_current, d_next);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "CUDA compute time: " << ms << " ms\n";

  // Copy the padded grid back, then unpad into the un-padded N x N image.
  CUDA_CHECK(cudaMemcpy(host_padded.data(), d_current, bytes, cudaMemcpyDeviceToHost));
  std::vector<unsigned char> host(static_cast<size_t>(N) * N);
  for (int i = 0; i < N; ++i) {
    std::memcpy(&host[i * N], &host_padded[(i + 1) * stride + 1],
                static_cast<size_t>(N));
  }

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
