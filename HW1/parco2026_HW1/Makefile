CC       = g++ -std=c++11
OMP_FLAG = -fopenmp
OPT      = -O2
FAST_OPT = -O3 -march=native

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
	$(CC) $(FAST_OPT) $(OMP_FLAG) -o $@ $^ $(FLAGS)

# False sharing experiment
false_sharing_exp: false_sharing_exp.cpp
	$(CC) $(OPT) $(OMP_FLAG) -o $@ $^ $(FLAGS)

# Generate default input files
inputs:
	python3 gen_input.py

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
	rm -f output*.dat
	rm -rf results/

.PHONY: all clean inputs check check_fast
