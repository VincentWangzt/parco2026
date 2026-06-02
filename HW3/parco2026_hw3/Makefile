CXX = g++ -std=c++11
MPICXX = mpicxx -std=c++11
NVCC = nvcc -std=c++11
EXES = serial parallel

all: $(EXES)

serial: serial.cpp
	$(CXX) -O2 -o $@ $^

parallel: parallel.cu
	$(NVCC) -O3 -o $@ $^

clean:
	rm -f $(EXES) *.o
