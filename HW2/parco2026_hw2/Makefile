CC = g++ -std=c++11
MPICC = mpic++ -std=c++11
# -O3 + native ISA (AVX-512 on EPYC 9754) so the compiler can vectorise the
# inner reduction. -funroll-loops hides the integer-multiply latency on the
# accumulator. -mprefer-vector-width=512 forces gcc to use the 512-bit zmm
# registers; without it gcc 8.x targets znver1's cost model and emits 128-bit
# xmm code even when AVX-512 is available, leaving ~3-4x throughput on the
# table for narrow integer types. Cluster build uses mpiicc; if mpiicc is not
# on PATH (local box) Makefile.local overrides MPICC -> mpicxx with the same
# flags.
FLAGS = -O3 -march=native -mprefer-vector-width=512 -funroll-loops -ftree-vectorize
EXES = serial parallel serial_timed

all: $(EXES)

serial: serial.cpp
	$(CC) -o $@ $^ $(FLAGS)

parallel: parallel.cpp
	$(MPICC) -o $@ $^ $(FLAGS)

# serial_timed is the serial algorithm wrapped in MPI_Wtime so the
# strong-scaling script can use the SAME clock as the parallel run.
# It is built with mpiicc only because it #include <mpi.h>; it still
# runs as a single process.
serial_timed: serial_timed.cpp
	$(MPICC) -o $@ $^ $(FLAGS)

clean:
	rm -f $(EXES) *.o
