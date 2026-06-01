CC = g++ -std=c++11
MPICC = mpic++ -std=c++11
# -O3 + so the compiler can vectorise the
# inner reduction. -funroll-loops hides the integer-multiply latency on the
# accumulator. 
FLAGS = -O3 -march=native -funroll-loops -ftree-vectorize
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
