#!/usr/bin/env python3
"""Visualise a Game-of-Life output file as a black/white PNG.

Usage: python3 visual.py life_N16_T4.txt life_N16_T4.png
"""
import re
import sys


def read_grid(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        rows, cols = int(header[0]), int(header[1])
        vals = []
        for line in f:
            vals.extend(int(t) for t in re.findall(r"[01]", line))
    return rows, cols, vals


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 visual.py <input.txt> <output.png>", file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    rows, cols, vals = read_grid(src)
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed (pip install pillow).", file=sys.stderr)
        return 2
    # Live = black (0), dead = white (255). Scale up so small grids are readable.
    scale = max(1, 512 // max(rows, cols))
    img = Image.new("L", (cols, rows), 255)
    px = img.load()
    for i in range(rows):
        for j in range(cols):
            px[j, i] = 0 if vals[i * cols + j] == 1 else 255
    img = img.resize((cols * scale, rows * scale), Image.NEAREST)
    img.save(dst)
    print(f"Wrote {dst} ({cols * scale}x{rows * scale})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
