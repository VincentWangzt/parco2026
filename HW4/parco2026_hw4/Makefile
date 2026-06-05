CXX = g++ -std=c++11
MPICXX = mpicxx -std=c++11
NVCC = nvcc -std=c++11

EXES = serial parallel_mpi parallel_cuda

all: $(EXES)

serial: serial.cpp
	$(CXX) -O2 -o $@ $^

parallel_mpi: parallel_mpi.cpp
	$(MPICXX) -O3 -o $@ $^

parallel_cuda: parallel_cuda.cu
	$(NVCC) -O3 -arch=sm_60 -o $@ $^

clean:
	rm -f $(EXES) *.o life_N*_T*.txt
