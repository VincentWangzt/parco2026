CC = g++ -std=c++11
MPICC = mpiicc -std=c++11
EXES = serial parallel

all: $(EXES)

serial: serial.cpp
	$(CC) -o $@ $^ $(FLAGS)

parallel: parallel.cpp
	$(MPICC) -o $@ $^ $(FLAGS)

clean:
	rm -f $(EXES) *.o
