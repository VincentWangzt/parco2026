/*
 * serial_timed.cpp — instrumented serial reference.
 *
 * Same algorithm as serial.cpp, but wrapped in MPI_Init/MPI_Wtime so that
 * the serial measurement uses the SAME clock and the SAME timing protocol
 * (Barrier + Wtime) as the parallel version. This is required by the
 * assignment ("不要使用 clock() / std::chrono"), and it eliminates a
 * systematic bias when comparing T_serial vs T_parallel.
 *
 * Run with EXACTLY ONE process:
 *   mpirun -np 1 ./serial_timed <N>
 *
 * This binary is built only by Makefile.local for local benchmarking.
 * It is not part of the assignment submission.
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

inline long long A_elem(long i, long j) {
  return (i * 7LL + j * 13LL) % 1000;
}

inline long long x_elem(long j) {
  return (j * 3LL + 1) % 100;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (size != 1 || rank != 0) {
    if (rank == 0)
      std::cerr << "serial_timed must be run with -np 1\n";
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  long N = (argc < 2) ? 32 : std::atol(argv[1]);
  std::cout << "Executing serial_timed with N = " << N << "\n";

  std::vector<long long> x(N);
  for (long j = 0; j < N; ++j) x[j] = x_elem(j);

  // Allocate A locally so the serial version pays for the same memory it
  // would otherwise touch — important to make the serial vs parallel
  // memory traffic comparable. (Initialising A inside the timed section
  // would be unfair, so we do it before the barrier, just like parallel.)
  std::vector<long long> A(static_cast<size_t>(N) * N);
  for (long i = 0; i < N; ++i) {
    long long* row = A.data() + static_cast<size_t>(i) * N;
    for (long j = 0; j < N; ++j) row[j] = A_elem(i, j);
  }

  std::vector<long long> y(N, 0);

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  for (long i = 0; i < N; ++i) {
    const long long* row = A.data() + static_cast<size_t>(i) * N;
    long long s = 0;
    for (long j = 0; j < N; ++j) s += row[j] * x[j];
    y[i] = s;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();
  std::cout << "time=" << (t1 - t0) << " s\n";

  // Write result file under a distinct name so it cannot accidentally
  // overwrite the parallel run's y_N<N>.txt.
  std::string filename = "y_N" + std::to_string(N) + "_serial.txt";
  std::ofstream ofs(filename);
  for (long i = 0; i < N; ++i) ofs << y[i] << "\n";
  std::cout << "Wrote result to " << filename << "\n";

  MPI_Finalize();
  return 0;
}
