# Foundations of Parallel Computing II, Spring 2026.
# Instructor: Chao Yang @ Peking University.
# Generate random input data for histogram computation.
#
# Parameters:
#   N        : Total number of data points (floating-point values).
#              Larger N means more computation and more visible parallel speedup.
#              Typical values: input1 uses a smaller N (e.g. 1e7) for quick validation,
#                              input2 uses a larger N (e.g. 1e8) for performance testing.
#   M        : Number of histogram bins. The range [min_val, max_val) is divided
#              evenly into M intervals. Smaller M (e.g. 16-256) makes false sharing
#              more pronounced and is better suited for comparison experiments.
#              Typical value: 256.
#   min_val  : Lower bound of the data range (data is uniformly distributed in [min_val, max_val)).
#   max_val  : Upper bound of the data range.
#   output   : Output file path (e.g. input1.dat).
#   seed     : Random seed for reproducibility. Default: 42.
#
# Output file format:
#   Line 1      : N M min_val max_val
#   Lines 2~N+1 : One floating-point number per line (8 decimal places)
#
# Default cases (run without arguments to generate both standard inputs):
#   input1.dat : N=10,000,000  M=256  [0.0, 1.0)  seed=42   (small case, for correctness check)
#   input2.dat : N=100,000,000 M=256  [0.0, 1.0)  seed=123  (large case, for performance testing)
#
# Usage: python gen_input.py <N> <M> <min_val> <max_val> <output_file> [seed]
#        python gen_input.py          # no args: generate default input1.dat and input2.dat

import numpy as np
import sys

# ── Default cases ─────────────────────────────────────────────────────────────
DEFAULT_CASES = [
    # (N,           M,   min_val, max_val, output_file,  seed)
    (10_000_000,   256,  0.0,     1.0,    "input1.dat",  42),   # small case
    (100_000_000,  256,  0.0,     1.0,    "input2.dat",  123),  # large case
]
# ─────────────────────────────────────────────────────────────────────────────

def gen_input(N, M, min_val, max_val, output_file, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.uniform(min_val, max_val, N)
    with open(output_file, 'w') as f:
        f.write(f"{N} {M} {min_val} {max_val}\n")
        for x in data:
            f.write(f"{x:.8f}\n")
    print(f"Generated N={N}, M={M}, range=[{min_val},{max_val}] -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: generate both default cases
        for (N, M, min_val, max_val, output, seed) in DEFAULT_CASES:
            gen_input(N, M, min_val, max_val, output, seed)
    elif len(sys.argv) < 6:
        print("Usage: python gen_input.py <N> <M> <min_val> <max_val> <output_file> [seed]")
        print("       python gen_input.py          # generate default input1.dat and input2.dat")
        sys.exit(1)
    else:
        N        = int(sys.argv[1])
        M        = int(sys.argv[2])
        min_val  = float(sys.argv[3])
        max_val  = float(sys.argv[4])
        output   = sys.argv[5]
        seed     = int(sys.argv[6]) if len(sys.argv) > 6 else 42
        gen_input(N, M, min_val, max_val, output, seed)
