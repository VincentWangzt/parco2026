#!/usr/bin/env python3
"""
Verifier for the Parallel Matrix-Vector Multiplication assignment.

The student's program (serial.cpp / parallel.cpp) writes a result file
    y_N<N>.txt
containing the N integers of y = A * x, one per line.

This script reads that file, recomputes the reference y using the SAME
deterministic formulas as the assignment, and reports any mismatches.

Examples
--------
    python3 verify.py y_N32.txt
    python3 verify.py y_N1000.txt
    python3 verify.py y_N8192.txt --N 8192

Exit codes
----------
    0  PASS
    1  FAIL (value mismatch or wrong length)
    2  ERROR (file not found or parse failure)
"""

import argparse
import os
import re
import sys


# -----------------------------------------------------------------------------
# Reference y = A * x.  MUST stay in sync with serial.cpp / parallel.cpp.
# -----------------------------------------------------------------------------
def compute_reference(N: int, chunk_rows: int = 1024):
    """Return the ground-truth y as a list of int.

    Uses a chunked numpy implementation so peak memory is
    chunk_rows * N * 8 bytes (e.g. 256 MB for N=32768).
    Falls back to pure Python if numpy is unavailable.
    """
    try:
        import numpy as np
        jj = np.arange(N, dtype=np.int64)
        x = (jj * 3 + 1) % 100
        y = np.empty(N, dtype=np.int64)
        for s in range(0, N, chunk_rows):
            e = min(s + chunk_rows, N)
            ii = np.arange(s, e, dtype=np.int64).reshape(-1, 1)
            A_chunk = (ii * 7 + jj * 13) % 1000
            y[s:e] = A_chunk @ x
        return y.tolist()
    except ImportError:
        x = [(j * 3 + 1) % 100 for j in range(N)]
        y = [0] * N
        for i in range(N):
            s = 0
            for j in range(N):
                s += ((i * 7 + j * 13) % 1000) * x[j]
            y[i] = s
        return y


# -----------------------------------------------------------------------------
# File reading.
# -----------------------------------------------------------------------------
def read_y(path: str):
    """Read y values from a text file. Tolerates blank lines and
    one-or-more-integers per line."""
    ys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for tok in re.findall(r"-?\d+", line):
                ys.append(int(tok))
    return ys


def infer_N(path: str, n_values: int):
    """Try to deduce N from the filename pattern y_N<int>.txt; otherwise
    fall back to the number of values read."""
    m = re.search(r"y_N(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return n_values


# -----------------------------------------------------------------------------
# Driver.
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Verify y_N<N>.txt against the reference y = A * x."
    )
    ap.add_argument("output", help="Path to the y_N<N>.txt file.")
    ap.add_argument(
        "--N",
        type=int,
        default=None,
        help="Override N (otherwise inferred from filename or value count).",
    )
    ap.add_argument(
        "--max-show",
        type=int,
        default=10,
        help="Maximum number of mismatches to display (default: 10).",
    )
    args = ap.parse_args()

    # --- read ---
    try:
        y_student = read_y(args.output)
    except FileNotFoundError:
        print(f"ERROR: file not found: {args.output}", file=sys.stderr)
        sys.exit(2)

    if not y_student:
        print(f"ERROR: no integers found in {args.output}", file=sys.stderr)
        sys.exit(2)

    N = args.N if args.N is not None else infer_N(args.output, len(y_student))

    # --- shape check ---
    if len(y_student) != N:
        print(f"FAIL: expected {N} values, got {len(y_student)}.")
        sys.exit(1)

    # --- value check ---
    y_ref = compute_reference(N)
    mismatches = [
        (i, y_student[i], y_ref[i])
        for i in range(N)
        if y_student[i] != y_ref[i]
    ]

    if not mismatches:
        print(f"PASS: all {N} elements of y match the reference.")
        if N >= 10:
            print(f"      y[:5]  = {y_ref[:5]}")
            print(f"      y[-5:] = {y_ref[-5:]}")
        else:
            print(f"      y      = {y_ref}")
        sys.exit(0)

    print(f"FAIL: {len(mismatches)} / {N} elements mismatch.")
    print(f"      showing first {min(len(mismatches), args.max_show)}:")
    for idx, got, want in mismatches[: args.max_show]:
        print(f"        y[{idx}] = {got}   (expected {want},   diff = {got - want})")
    sys.exit(1)


if __name__ == "__main__":
    main()
