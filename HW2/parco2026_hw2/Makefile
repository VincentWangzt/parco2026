CC = g++ -std=c++11
MPICC = mpiicc -std=c++11
FLAGS = -O3
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
