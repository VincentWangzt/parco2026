/*
 * Foundations of Parallel Computing II.
 * MPI implementation of Conway's Game of Life.
 *
 * Decomposition: 1-D row partition. Each rank owns `local_rows` consecutive
 * rows of an N x N grid plus 2 ghost rows (top + bottom) and 2 ghost columns
 * (left + right) so the inner stencil never has to bounds-check. Total local
 * buffer is (local_rows + 2) x (N + 2) bytes; the 4 outer borders are kept at
 * zero (fixed dead boundary).
 *
 * Each step:
 *   1. Exchange top/bottom interior rows with up/down neighbors via
 *      MPI_Sendrecv (length N, starting at column 1 of the padded buffer).
 *   2. Update local rows from `current` into `next` with a branch-free 8-add
 *      neighbour reduction.
 *   3. Swap buffers.
 * After T steps rank 0 gathers the full grid via MPI_Gatherv (one row at a
 * time would be slow; instead we pack each rank's local rows into a tight
 * contiguous buffer first) and writes life_N<N>_T<T>.txt.
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
  std::vector<int> counts(size), offsets(size);  // in cells (for Gatherv)
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

  // Padded local buffer:
  //   stride = N + 2  (col 0 and col N+1 are zero ghost cols)
  //   height = local_rows + 2  (row 0 and row local_rows+1 are ghost rows)
  // Real data lives in rows [1, local_rows], cols [1, N].
  size_t stride = static_cast<size_t>(N + 2);
  size_t local_h = static_cast<size_t>(local_rows + 2);
  std::vector<unsigned char> current(local_h * stride, 0);
  std::vector<unsigned char> next(local_h * stride, 0);

  // Initialize local rows using the deterministic formula (offset by 1 col).
  for (int i = 0; i < local_rows; ++i) {
    int gi = row_start + i;
    unsigned char* row = &current[(i + 1) * stride + 1];
    for (int j = 0; j < N; ++j) {
      row[j] = static_cast<unsigned char>(init_cell(gi, j));
    }
  }

  int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

  MPI_Barrier(MPI_COMM_WORLD);
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < T; ++step) {
    // Exchange ghost rows (only the N real cells; ghost cols stay zero).
    unsigned char* top_row = &current[1 * stride + 1];
    unsigned char* bot_row = &current[local_rows * stride + 1];
    unsigned char* top_ghost = &current[0 * stride + 1];
    unsigned char* bot_ghost = &current[(local_rows + 1) * stride + 1];

    MPI_Sendrecv(top_row, N, MPI_UNSIGNED_CHAR, up, 0,
                 bot_ghost, N, MPI_UNSIGNED_CHAR, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(bot_row, N, MPI_UNSIGNED_CHAR, down, 1,
                 top_ghost, N, MPI_UNSIGNED_CHAR, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (up == MPI_PROC_NULL)   std::memset(top_ghost, 0, N);
    if (down == MPI_PROC_NULL) std::memset(bot_ghost, 0, N);

    // Update local rows. With ghost cols pinned at zero we can drop the
    // jl/jr branches: the loop body is a flat 8-add reduction the compiler
    // can vectorise.
    for (int i = 1; i <= local_rows; ++i) {
      const unsigned char* up_r = &current[(i - 1) * stride];
      const unsigned char* mid  = &current[i * stride];
      const unsigned char* dn_r = &current[(i + 1) * stride];
      unsigned char* out = &next[i * stride];
      for (int j = 1; j <= N; ++j) {
        int cnt = up_r[j - 1] + up_r[j] + up_r[j + 1]
                + mid[j - 1]            + mid[j + 1]
                + dn_r[j - 1] + dn_r[j] + dn_r[j + 1];
        int alive = mid[j];
        out[j] = (alive ? (cnt == 2 || cnt == 3) : (cnt == 3)) ? 1 : 0;
      }
    }
    current.swap(next);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Pack local interior (skip ghost rows AND ghost cols) into a tight buffer
  // for Gatherv; the destination grid on rank 0 is the un-padded N x N image.
  std::vector<unsigned char> local_packed(static_cast<size_t>(local_rows) * N);
  for (int i = 0; i < local_rows; ++i) {
    std::memcpy(&local_packed[i * N],
                &current[(i + 1) * stride + 1],
                static_cast<size_t>(N));
  }

  std::vector<unsigned char> full;
  if (rank == 0) full.resize(static_cast<size_t>(N) * N);
  MPI_Gatherv(local_packed.data(), local_rows * N, MPI_UNSIGNED_CHAR,
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
