#!/usr/bin/env python3
"""
Verifier for the Game of Life assignment.

The program writes:
    life_N<N>_T<T>.txt
whose first line is "N N", followed by the final N x N grid.
"""

import os
import re
import sys


def init_cell(i: int, j: int) -> int:
    return 1 if ((i * 131 + j * 17 + 7) % 100) < 30 else 0


def parse_name(path: str):
    m = re.search(r"life_N(\d+)_T(\d+)", os.path.basename(path))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def read_grid(path: str):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) != 2:
            raise ValueError("first line must be: N N")
        rows, cols = map(int, header)
        values = []
        for line in f:
            values.extend(int(tok) for tok in re.findall(r"[01]", line))
    if rows != cols:
        raise ValueError("only square grids are expected")
    if len(values) != rows * cols:
        raise ValueError(f"expected {rows * cols} cells, got {len(values)}")
    return rows, values


def step(grid, n: int):
    nxt = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            cnt = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        cnt += grid[ni * n + nj]
            alive = grid[i * n + j]
            nxt[i * n + j] = 1 if ((alive and cnt in (2, 3)) or ((not alive) and cnt == 3)) else 0
    return nxt


def reference(n: int, t: int):
    grid = [init_cell(i, j) for i in range(n) for j in range(n)]
    for _ in range(t):
        grid = step(grid, n)
    return grid


def main() -> int:
    if len(sys.argv) not in (2, 4):
        print("Usage: python3 verify.py life_N<N>_T<T>.txt [N T]", file=sys.stderr)
        return 2

    path = sys.argv[1]
    try:
        n_file, got = read_grid(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if len(sys.argv) == 4:
        n = int(sys.argv[2])
        t = int(sys.argv[3])
    else:
        n, t = parse_name(path)
        if n is None:
            print("ERROR: cannot infer N and T from filename; pass them explicitly.", file=sys.stderr)
            return 2

    if n != n_file:
        print(f"FAIL: filename says N={n}, file header says N={n_file}.")
        return 1

    ref = reference(n, t)
    mismatches = [(k, got[k], ref[k]) for k in range(n * n) if got[k] != ref[k]]
    if mismatches:
        print(f"FAIL: {len(mismatches)} cells mismatch; showing first 10:")
        for k, value, expected in mismatches[:10]:
            print(f"  cell({k // n}, {k % n}) = {value} (expected {expected})")
        return 1

    live = sum(got)
    print(f"PASS: final grid matches the reference for N={n}, T={t}.")
    print(f"      live cells = {live}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
