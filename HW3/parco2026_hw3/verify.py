#!/usr/bin/env python3
"""
Verifier for the CUDA prefix-sum assignment.

The program writes:
    scan_N<N>.txt
containing one integer per line.
"""

import os
import re
import sys


def x_elem(i: int) -> int:
    return (i * 17 + 13) % 1000


def read_values(path: str):
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for tok in re.findall(r"-?\d+", line):
                values.append(int(tok))
    return values


def infer_n(path: str, count: int) -> int:
    m = re.search(r"scan_N(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return count


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 verify.py scan_N<N>.txt [N]", file=sys.stderr)
        return 2

    path = sys.argv[1]
    try:
        got = read_values(path)
    except FileNotFoundError:
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    if not got:
        print(f"ERROR: no integers found in {path}", file=sys.stderr)
        return 2

    n = int(sys.argv[2]) if len(sys.argv) == 3 else infer_n(path, len(got))
    if len(got) != n:
        print(f"FAIL: expected {n} values, got {len(got)}.")
        return 1

    mismatches = []
    running = 0
    for i in range(n):
        running += x_elem(i)
        if got[i] != running:
            mismatches.append((i, got[i], running))
            if len(mismatches) >= 10:
                break

    if mismatches:
        print("FAIL: prefix sum mismatch.")
        for idx, value, expected in mismatches:
            print(f"  y[{idx}] = {value} (expected {expected}, diff = {value - expected})")
        return 1

    print(f"PASS: all {n} elements match the reference prefix sum.")
    print(f"      y[0] = {got[0]}, y[-1] = {got[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
