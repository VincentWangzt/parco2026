# Foundations of Parallel Computing II, Spring 2026.
# Instructor: Chao Yang @ Peking University.
# Verify correctness of histogram output.
#
# Usage: python check.py <input_file> <output_file>

import numpy as np
import sys

def read_input(filename):
    with open(filename, 'r') as f:
        N, M, min_val, max_val = f.readline().split()
        N, M = int(N), int(M)
        min_val, max_val = float(min_val), float(max_val)
        data = np.array([float(f.readline()) for _ in range(N)])
    return data, M, min_val, max_val

def read_output(filename):
    with open(filename, 'r') as f:
        hist = np.array([int(line.strip()) for line in f if line.strip()])
    return hist

if len(sys.argv) != 3:
    print("Usage: python check.py <input_file> <output_file>")
    sys.exit(1)

data, M, min_val, max_val = read_input(sys.argv[1])
hist = read_output(sys.argv[2])

# Reference histogram using numpy
ref_hist, _ = np.histogram(data, bins=M, range=(min_val, max_val))

print(f"Number of bins      : {len(hist)} (expected {M})")
print(f"Total count         : {np.sum(hist)} (expected {len(data)})")
print(f"Count correct       : {np.sum(hist) == len(data)}")
print(f"Matches reference   : {np.array_equal(hist, ref_hist)}")

if not np.array_equal(hist, ref_hist):
    diff = np.abs(hist.astype(np.int64) - ref_hist.astype(np.int64))
    print(f"Max bin difference  : {np.max(diff)}")
    print(f"Number of wrong bins: {np.sum(diff > 0)}")
else:
    print("Result is CORRECT.")
