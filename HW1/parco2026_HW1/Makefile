CC       = g++ -std=c++11
OMP_FLAG = -fopenmp
OPT      = -O3 -march=native

# Executables
SERIAL       = serial
HISTOGRAM    = histogram_omp
FAST         = histogram_fast
FALSESHARE   = false_sharing_exp

all: $(SERIAL) $(HISTOGRAM) $(FAST) $(FALSESHARE)

# Serial: NO -fopenmp, uses <chrono> for timing
serial: serial.cpp
	$(CC) $(OPT) -o $@ $^ $(FLAGS)

# Multi-strategy OpenMP version
histogram_omp: histogram_omp.cpp
	$(CC) $(OPT) $(OMP_FLAG) -o $@ $^ $(FLAGS)

# Optimized fast version
histogram_fast: histogram_fast.cpp
	$(CC) $(OPT) $(OMP_FLAG) -o $@ $^ $(FLAGS)

# False sharing experiment
false_sharing_exp: false_sharing_exp.cpp
	$(CC) $(OPT) $(OMP_FLAG) -o $@ $^ $(FLAGS)

# Generate default input files
inputs:
	python3 gen_input.py

# Generate binary input files (text -> binary conversion)
inputs-bin: inputs
	python3 convert_to_binary.py input1.dat input1.bin
	python3 convert_to_binary.py input2.dat input2.bin

# Quick correctness check
check: $(HISTOGRAM)
	@echo "=== Checking all strategies ==="
	@for s in serial atomic critical private padded reduction; do \
		echo -n "  $$s: "; \
		OMP_NUM_THREADS=4 ./$(HISTOGRAM) $$s input1.dat output_check.dat 2>/dev/null; \
		python3 check.py input1.dat output_check.dat | tail -1; \
	done
	@rm -f output_check.dat

check_fast: $(FAST)
	@echo "=== Checking histogram_fast ==="
	OMP_NUM_THREADS=4 ./$(FAST) input1.dat output_fast_check.dat 2>/dev/null
	python3 check.py input1.dat output_fast_check.dat
	@rm -f output_fast_check.dat

clean:
	rm -f $(SERIAL) $(HISTOGRAM) $(FAST) $(FALSESHARE) *.o
	rm -f output*.dat *.bin
	rm -rf results/

clean-bench:
	./cleanup.sh

clean-all:
	./cleanup.sh --all

.PHONY: all clean clean-bench clean-all inputs inputs-bin check check_fast
