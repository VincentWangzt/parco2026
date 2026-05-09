# Foundations of Parallel Computing II, Spring 2026.
# Instructor: Chao Yang @ Peking University.
# Convert between text and binary formats for histogram I/O.
#
# Forward (text -> binary input):
#   python convert_to_binary.py input1.dat input1.bin
#
# Reverse (binary output -> text output):
#   python convert_to_binary.py --reverse output.bin output.dat
#
# Binary input format (little-endian):
#   [N: int32][M: int32][min_val: float64][max_val: float64][data: N x float64]
#
# Binary output format (little-endian):
#   [hist: M x int32]

import numpy as np
import struct
import sys
import os


def text_to_binary_input(src, dst):
    """Convert text input file (.dat) to binary input file (.bin).

    Text format:
        Line 1: N M min_val max_val
        Lines 2..N+1: one float per line

    Binary format (little-endian):
        [N:int32][M:int32][min_val:f64][max_val:f64][data: N x f64]
    """
    with open(src, 'r') as f:
        header = f.readline().split()
        N, M = int(header[0]), int(header[1])
        min_val, max_val = float(header[2]), float(header[3])
        # Read all remaining data at once with numpy for speed
        data = np.loadtxt(f, dtype=np.float64, max_rows=N)

    with open(dst, 'wb') as f:
        f.write(struct.pack('<ii', N, M))
        f.write(struct.pack('<dd', min_val, max_val))
        data.tofile(f)

    expected_size = 4 + 4 + 8 + 8 + 8 * N
    actual_size = os.path.getsize(dst)
    assert actual_size == expected_size, \
        f"Size mismatch: expected {expected_size}, got {actual_size}"
    print(f"Converted {src} -> {dst} (N={N}, M={M}, {actual_size} bytes)")


def binary_to_text_output(src, dst):
    """Convert binary histogram output (.bin) to text output (.dat).

    Binary format: M x int32 (little-endian)
    Text format: one integer per line
    """
    hist = np.fromfile(src, dtype=np.int32)
    with open(dst, 'w') as f:
        for v in hist:
            f.write(f"{v}\n")
    print(f"Converted {src} -> {dst} ({len(hist)} bins)")


def print_usage():
    print("Usage:")
    print("  python convert_to_binary.py <input.dat> <output.bin>")
    print("      Convert text input to binary input format.")
    print("")
    print("  python convert_to_binary.py --reverse <input.bin> <output.dat>")
    print("      Convert binary histogram output to text format.")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()

    if sys.argv[1] == "--reverse":
        if len(sys.argv) != 4:
            print_usage()
        binary_to_text_output(sys.argv[2], sys.argv[3])
    else:
        if len(sys.argv) != 3:
            print_usage()
        text_to_binary_input(sys.argv[1], sys.argv[2])
