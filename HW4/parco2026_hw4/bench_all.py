#!/usr/bin/env python3
"""Run all variants on cases S/M/L, 11 reps each, record median compute time.

Outputs results.txt summarising medians and speedups vs serial."""

import os, re, statistics, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

CASES = [("S", 16, 4), ("M", 257, 100), ("L", 1024, 200)]
RUNS = 11
TIME_RE = re.compile(r"compute time:\s*([\d.eE+-]+)\s*ms")


def time_runs(label, cmd):
    times = []
    for _ in range(RUNS):
        out = subprocess.run(cmd, capture_output=True, text=True)
        m = TIME_RE.search(out.stdout)
        if not m:
            print(f"[{label}] no time line:\n{out.stdout}\n{out.stderr}", file=sys.stderr)
            sys.exit(1)
        times.append(float(m.group(1)))
    return times


def variants_for(N, T):
    return [
        ("serial",         ["./serial", str(N), str(T)]),
        ("mpi_np4",        ["mpirun", "--allow-run-as-root", "-np", "4",  "./parallel_mpi", str(N), str(T)]),
        ("mpi_np16",       ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "16", "./parallel_mpi", str(N), str(T)]),
        ("cuda",           ["./parallel_cuda", str(N), str(T)]),
    ]


def main():
    out_lines = []
    out_lines.append(f"# Game-of-Life benchmark — median of {RUNS} runs (ms)")
    for tag, N, T in CASES:
        out_lines.append(f"\n## Case {tag} : N={N}, T={T}")
        results = {}
        for label, cmd in variants_for(N, T):
            ts = time_runs(label, cmd)
            med = statistics.median(ts)
            mn, mx = min(ts), max(ts)
            results[label] = med
            out_lines.append(f"  {label:<14} runs = {['%.3f' % t for t in ts]}")
            out_lines.append(f"  {label:<14} median = {med:8.3f} ms  (min {mn:.3f}, max {mx:.3f})")
        base = results["serial"]
        out_lines.append(f"  {'-' * 60}")
        out_lines.append(f"  {'variant':<14}{'median(ms)':>14}{'speedup':>14}")
        for label, med in results.items():
            sp = base / med if med > 0 else float("inf")
            out_lines.append(f"  {label:<14}{med:>14.3f}{sp:>13.2f}x")

    text = "\n".join(out_lines)
    print(text)
    with open("results.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
