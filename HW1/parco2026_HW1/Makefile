CC  = g++ -std=c++11
OBJ = *.o
EXE = serial

all: ${EXE}

serial: serial.cpp
	$(CC) -o $@ $^ $(FLAGS)

clean:
	rm -f $(OBJ) $(EXE)
