CXX = g++ -std=c++11
MPICXX = mpicxx -std=c++11
NVCC = nvcc -std=c++11

# OpenMPI installed in /usr/lib64/openmpi.
MPI_HOME ?= /usr/lib64/openmpi
MPI_INC  ?= /usr/include/openmpi-x86_64
MPI_LIB  ?= $(MPI_HOME)/lib

EXES = serial parallel_mpi parallel_cuda parallel_mpi_cuda

all: $(EXES)

serial: serial.cpp
	$(CXX) -O2 -o $@ $^

parallel_mpi: parallel_mpi.cpp
	$(MPICXX) -O3 -o $@ $^

parallel_cuda: parallel_cuda.cu
	$(NVCC) -O3 -arch=sm_60 -o $@ $^

parallel_mpi_cuda: parallel_mpi_cuda.cu
	$(NVCC) -O3 -arch=sm_60 -ccbin g++ \
		-Xcompiler -O3 -Xcompiler -I$(MPI_INC) \
		-I$(MPI_INC) -L$(MPI_LIB) \
		-o $@ $^ -lmpi

clean:
	rm -f $(EXES) *.o life_N*_T*.txt
