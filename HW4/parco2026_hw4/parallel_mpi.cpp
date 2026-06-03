/*
 * Foundations of Parallel Computing II.
 * MPI implementation of Conway's Game of Life.
 *
 * Decomposition: 1-D row partition. Each rank owns `local_rows` consecutive
 * rows of an N x N grid plus 2 ghost rows (top + bottom) that hold the
 * neighboring ranks' boundary rows (or zeros at the global boundary).
 *
 * Each step:
 *   1. Exchange top/bottom rows with up/down neighbors via MPI_Sendrecv.
 *   2. Update local rows from `current` into `next`.
 *   3. Swap buffers.
 * After T steps rank 0 gathers the full grid via MPI_Gatherv and writes it
 * to life_N<N>_T<T>.txt in the same format as serial.cpp.
 *
 * Initial grid: grid[i][j] = (((i*131 + j*17 + 7) % 100) < 30)
 * Boundary: cells outside the grid are dead (fixed dead boundary).
 *
 * Usage: mpirun -np P ./parallel_mpi <N> <T>
 */

#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

inline int init_cell(int i, int j) {
  return ((i * 131 + j * 17 + 7) % 100) < 30 ? 1 : 0;
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

  if (rank == 0) {
    std::cout << "Executing MPI Game of Life with N = " << N
              << ", T = " << T << ", P = " << size << "\n";
  }

  // Row partition: rank r owns rows [row_start, row_start + local_rows).
  // Distribute the remainder to the first (N % size) ranks.
  std::vector<int> counts(size), offsets(size);  // in cells
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

  // Local buffers carry 2 ghost rows: index 0 (top) and index local_rows+1 (bottom).
  // Total height = local_rows + 2.
  size_t stride = static_cast<size_t>(N);
  size_t local_h = static_cast<size_t>(local_rows + 2);
  std::vector<unsigned char> current(local_h * stride, 0);
  std::vector<unsigned char> next(local_h * stride, 0);

  // Initialize local rows using the deterministic formula.
  for (int i = 0; i < local_rows; ++i) {
    int gi = row_start + i;
    for (int j = 0; j < N; ++j) {
      current[(i + 1) * stride + j] = static_cast<unsigned char>(init_cell(gi, j));
    }
  }

  int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

  MPI_Barrier(MPI_COMM_WORLD);
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < T; ++step) {
    // Exchange ghost rows. Send my top row up, receive bottom ghost from down;
    // send my bottom row down, receive top ghost from up.
    unsigned char* top_row = &current[1 * stride];
    unsigned char* bot_row = &current[local_rows * stride];
    unsigned char* top_ghost = &current[0];
    unsigned char* bot_ghost = &current[(local_rows + 1) * stride];

    MPI_Sendrecv(top_row, N, MPI_UNSIGNED_CHAR, up, 0,
                 bot_ghost, N, MPI_UNSIGNED_CHAR, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(bot_row, N, MPI_UNSIGNED_CHAR, down, 1,
                 top_ghost, N, MPI_UNSIGNED_CHAR, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (up == MPI_PROC_NULL)   std::memset(top_ghost, 0, N);
    if (down == MPI_PROC_NULL) std::memset(bot_ghost, 0, N);

    // Update local rows. Row index in local buffer is i+1 (i in [0, local_rows)).
    // For each cell sum the 8 neighbors honoring fixed dead boundary on left/right.
    for (int i = 1; i <= local_rows; ++i) {
      const unsigned char* up_r = &current[(i - 1) * stride];
      const unsigned char* mid  = &current[i * stride];
      const unsigned char* dn_r = &current[(i + 1) * stride];
      unsigned char* out = &next[i * stride];
      for (int j = 0; j < N; ++j) {
        int jl = j - 1, jr = j + 1;
        int cnt = 0;
        if (jl >= 0) cnt += up_r[jl] + mid[jl] + dn_r[jl];
        if (jr <  N) cnt += up_r[jr] + mid[jr] + dn_r[jr];
        cnt += up_r[j] + dn_r[j];
        int alive = mid[j];
        out[j] = (alive ? (cnt == 2 || cnt == 3) : (cnt == 3)) ? 1 : 0;
      }
    }
    current.swap(next);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Gather full grid on rank 0 (skip ghost rows).
  std::vector<unsigned char> full;
  if (rank == 0) full.resize(static_cast<size_t>(N) * N);
  MPI_Gatherv(&current[stride], local_rows * N, MPI_UNSIGNED_CHAR,
              rank == 0 ? full.data() : nullptr,
              counts.data(), offsets.data(), MPI_UNSIGNED_CHAR,
              0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "MPI compute time: " << ms << " ms\n";
    write_grid(full, N, T);
    if (N <= 32) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          std::cout << static_cast<int>(full[i * N + j]) << (j + 1 == N ? '\n' : ' ');
        }
      }
    }
  }

  MPI_Finalize();
  return 0;
}
