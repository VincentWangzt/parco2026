/*
 * Foundations of Parallel Computing II.
 * Serial implementation of Conway's Game of Life.
 *
 * Initial state:
 *   grid[i][j] = (((i * 131 + j * 17 + 7) % 100) < 30)
 *
 * Boundary condition: cells outside the grid are dead.
 *
 * Usage: ./serial <N> <T>
 */

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

inline int init_cell(int i, int j) {
  return ((i * 131 + j * 17 + 7) % 100) < 30 ? 1 : 0;
}

inline int idx(int i, int j, int N) {
  return i * N + j;
}

int live_neighbors(const std::vector<unsigned char>& grid, int i, int j, int N) {
  int count = 0;
  for (int di = -1; di <= 1; ++di) {
    for (int dj = -1; dj <= 1; ++dj) {
      if (di == 0 && dj == 0) continue;
      int ni = i + di;
      int nj = j + dj;
      if (ni >= 0 && ni < N && nj >= 0 && nj < N) {
        count += grid[idx(ni, nj, N)];
      }
    }
  }
  return count;
}

void write_grid(const std::vector<unsigned char>& grid, int N, int T) {
  std::string filename = "life_N" + std::to_string(N) + "_T" + std::to_string(T) + ".txt";
  std::ofstream ofs(filename);
  if (!ofs) {
    std::cerr << "ERROR: cannot open output file " << filename << "\n";
    std::exit(1);
  }
  ofs << N << " " << N << "\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      ofs << static_cast<int>(grid[idx(i, j, N)]);
      if (j + 1 < N) ofs << " ";
    }
    ofs << "\n";
  }
  std::cout << "Wrote result to " << filename << "\n";
}

int main(int argc, char* argv[]) {
  int N = 16;
  int T = 4;
  if (argc >= 2) N = std::atoi(argv[1]);
  if (argc >= 3) T = std::atoi(argv[2]);

  std::cout << "Executing serial Game of Life with N = " << N << ", T = " << T << "\n";

  std::vector<unsigned char> current(N * N);
  std::vector<unsigned char> next(N * N, 0);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      current[idx(i, j, N)] = static_cast<unsigned char>(init_cell(i, j));
    }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int step = 0; step < T; ++step) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        int alive = current[idx(i, j, N)];
        int cnt = live_neighbors(current, i, j, N);
        next[idx(i, j, N)] = (alive ? (cnt == 2 || cnt == 3) : (cnt == 3)) ? 1 : 0;
      }
    }
    current.swap(next);
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "Serial compute time: " << ms << " ms\n";

  write_grid(current, N, T);

  if (N <= 32) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        std::cout << static_cast<int>(current[idx(i, j, N)]) << (j + 1 == N ? '\n' : ' ');
      }
    }
  }

  return 0;
}
