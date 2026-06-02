CXX = g++ -std=c++11
MPICXX = mpicxx -std=c++11
NVCC = nvcc -std=c++11 -Xcompiler
EXES = serial

all: $(EXES)

serial: serial.cpp
	$(CXX) -O2 -o $@ $^

clean:
	rm -f $(EXES) *.o
