CC       = g++ -std=c++11
OMP_FLAG = -fopenmp
OPT      = -O3 -march=native

# ── OpenMP 4.5+ array reduction detection ────────────────────────────────────
# Try to compile a minimal snippet using reduction(+:arr[:n]).
# If the compiler supports it, define HAS_OMP45_REDUCTION so the code
# conditionally includes the reduction strategy.
OMP45_TEST := $(shell printf '%s\n' '\#include <omp.h>' \
  'int main(){int a[4]={}; int n=4;' \
  '_Pragma("omp parallel for reduction(+:a[:n])")' \
  'for(int i=0;i<n;i++) a[i]++; return 0;}' | \
  $(CC) $(OMP_FLAG) -x c++ -o /dev/null - 2>/dev/null && echo yes || echo no)

ifeq ($(OMP45_TEST),yes)
  OMP45_FLAGS = -DHAS_OMP45_REDUCTION
else
  OMP45_FLAGS =
endif

# Executables
SERIAL       = serial
HISTOGRAM    = histogram_omp
FAST         = histogram_fast
FAST_DIV     = histogram_fast_div
FALSESHARE   = false_sharing_exp

all: $(SERIAL) $(HISTOGRAM) $(FAST) $(FAST_DIV) $(FALSESHARE)
	@echo "[INFO] OpenMP 4.5+ array reduction: $(OMP45_TEST)"

# Serial: NO -fopenmp, uses <chrono> for timing
serial: serial.cpp
	$(CC) $(OPT) -o $@ $^ $(FLAGS)

# Multi-strategy OpenMP version
histogram_omp: histogram_omp.cpp
	$(CC) $(OPT) $(OMP_FLAG) $(OMP45_FLAGS) -o $@ $^ $(FLAGS)

# Optimized fast version (reciprocal multiply)
histogram_fast: histogram_fast.cpp
	$(CC) $(OPT) $(OMP_FLAG) -o $@ $^ $(FLAGS)

# Optimized fast version (standard division)
histogram_fast_div: histogram_fast_div.cpp
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

# Quick correctness check (auto-detects available strategies)
check: $(HISTOGRAM)
	@echo "=== Checking all strategies ==="
	@for s in $$(./$(HISTOGRAM) 2>&1 | sed -n 's/.*Strategies: *//p'); do \
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

check_fast_div: $(FAST_DIV)
	@echo "=== Checking histogram_fast_div ==="
	OMP_NUM_THREADS=4 ./$(FAST_DIV) input1.dat output_fast_div_check.dat 2>/dev/null
	python3 check.py input1.dat output_fast_div_check.dat
	@rm -f output_fast_div_check.dat

clean:
	rm -f $(SERIAL) $(HISTOGRAM) $(FAST) $(FAST_DIV) $(FALSESHARE) *.o
	rm -f output*.dat *.bin
	rm -rf results/

clean-bench:
	./cleanup.sh

clean-all:
	./cleanup.sh --all

.PHONY: all clean clean-bench clean-all inputs inputs-bin check check_fast check_fast_div
