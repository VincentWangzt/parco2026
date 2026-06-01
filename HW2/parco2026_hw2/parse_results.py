#!/usr/bin/env python3
"""parse_results.py — turn a bench.slurm log into the report tables.

Pipeline:
  cluster> sbatch bench.slurm                # produces bench.<id>.out
  local>   scp cluster:bench.<id>.out .
  local>   python3 parse_results.py bench.<id>.out

Log format (what bench.slurm writes):

    [bench] <exe>,<N>,<p>,<run>
    ...lines from ./serial_timed or ./parallel ...
    time=<sec> s

Every `[bench]` tag is paired with the FIRST `time=` line that follows
it. Lines without a tag (Phase 0 correctness output, banners, verify.py
output, etc.) are ignored, so the SLURM log can carry anything else
without confusing the parser.

The script:
  1. Walks the file, harvests one row per `[bench] … / time=` pair.
  2. Writes bench_results.csv (one row per measurement).
  3. Prints a tidy table grouped by (exe, N, p) with min/median/mean/max.
  4. Prints the case-L speedup line and the strong-scaling table that go
     verbatim into report.md.
  5. If matplotlib is available, regenerates figs/scaling_speedup.png and
     figs/scaling_efficiency.png.

Idempotent: rerunning overwrites bench_results.csv and the figs.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
from collections import defaultdict


TAG_RE  = re.compile(r"^\[bench\]\s+(?P<exe>\w+),(?P<N>\d+),(?P<p>\d+),(?P<run>\d+)\s*$")
TIME_RE = re.compile(r"^time=(?P<t>[0-9.eE+-]+)\s*s")


def parse_log(path: str):
    """Read a bench.slurm log and return a list of measurement rows.

    A row is emitted for every `[bench] …` tag paired with the next
    `time=` line. Anything else in the log is ignored.
    """
    rows = []
    pending = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            mt = TAG_RE.match(line)
            if mt:
                if pending is not None:
                    print(f"WARN: no time= for {pending}", file=sys.stderr)
                pending = mt.groupdict()
                continue
            if pending is not None:
                mtim = TIME_RE.match(line)
                if mtim:
                    rows.append({
                        "exe":    pending["exe"],
                        "N":      int(pending["N"]),
                        "p":      int(pending["p"]),
                        "run":    int(pending["run"]),
                        "time_s": float(mtim["t"]),
                    })
                    pending = None
    return rows


def write_csv(rows, path="bench_results.csv"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exe", "N", "p", "run", "time_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}  ({len(rows)} rows)")


def summarize(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["exe"], r["N"], r["p"])].append(r["time_s"])

    # raw table
    print()
    print(f"{'exe':<10}{'N':>8}{'p':>4}{'runs':>6}"
          f"{'min':>12}{'median':>12}{'mean':>12}{'max':>12}")
    for k in sorted(g.keys()):
        ts = g[k]
        print(f"{k[0]:<10}{k[1]:>8}{k[2]:>4}{len(ts):>6}"
              f"{min(ts):>12.6f}{statistics.median(ts):>12.6f}"
              f"{statistics.mean(ts):>12.6f}{max(ts):>12.6f}")

    # case-L speedup
    print()
    print("=== Case L speedup (median) ===")
    for (exe, N, p) in sorted(g.keys()):
        if exe != "serial":
            continue
        par_key = ("parallel", N, 16)
        if par_key not in g:
            continue
        ts_ser = statistics.median(g[(exe, N, p)])
        ts_par = statistics.median(g[par_key])
        speedup = ts_ser / ts_par if ts_par > 0 else float("inf")
        print(f"  N={N}: T_serial={ts_ser:.4f}s  "
              f"T_parallel(p=16)={ts_par:.4f}s  "
              f"S={speedup:.3f}x  E={speedup/16:.3f}")

    # strong scaling
    print()
    print("=== Strong scaling (median) ===")
    Ns = sorted({n for (e, n, p) in g.keys() if e == "parallel"})
    for N in Ns:
        ps = sorted(p for (e, nn, p) in g.keys() if e == "parallel" and nn == N)
        if (("parallel", N, 1) not in g) or len(ps) < 2:
            continue
        t1 = statistics.median(g[("parallel", N, 1)])
        print(f"--- N = {N} ---")
        print(f"  {'p':>4}{'T(p)':>12}{'S(p)':>10}{'E(p)':>10}")
        for p in ps:
            tp = statistics.median(g[("parallel", N, p)])
            s = t1 / tp
            e = s / p
            print(f"  {p:>4}{tp:>12.6f}{s:>10.3f}{e:>10.3f}")
    return g


def make_plots(g):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots", file=sys.stderr)
        return

    Ns = sorted({n for (e, n, p) in g.keys()
                 if e == "parallel" and ("parallel", n, 1) in g})
    if not Ns:
        return
    ps_all = sorted({p for (e, n, p) in g.keys() if e == "parallel"})
    os.makedirs("figs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    markers = ["o", "s", "^", "D", "v"]
    for i, N in enumerate(Ns):
        ps = sorted(p for (e, nn, p) in g.keys() if e == "parallel" and nn == N)
        t1 = statistics.median(g[("parallel", N, 1)])
        sp = [t1 / statistics.median(g[("parallel", N, p)]) for p in ps]
        ax.plot(ps, sp, marker=markers[i % len(markers)], label=f"measured, N={N}")
    ax.plot(ps_all, ps_all, "k--", label="ideal: S = p")
    ax.set_xlabel("number of MPI processes p")
    ax.set_ylabel("speedup S(p) = T(1) / T(p)")
    ax.set_title("Strong-scaling speedup")
    ax.set_xticks(ps_all)
    ax.set_yticks(ps_all)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figs/scaling_speedup.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, N in enumerate(Ns):
        ps = sorted(p for (e, nn, p) in g.keys() if e == "parallel" and nn == N)
        t1 = statistics.median(g[("parallel", N, 1)])
        eff = [t1 / statistics.median(g[("parallel", N, p)]) / p for p in ps]
        ax.plot(ps, eff, marker=markers[i % len(markers)], label=f"measured, N={N}")
    ax.axhline(1.0, color="k", linestyle="--", label="ideal: E = 1")
    ax.set_xlabel("number of MPI processes p")
    ax.set_ylabel("efficiency E(p) = S(p) / p")
    ax.set_title("Strong-scaling efficiency")
    ax.set_xticks(ps_all)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figs/scaling_efficiency.png", dpi=150)
    plt.close(fig)
    print("wrote figs/scaling_speedup.png and figs/scaling_efficiency.png")


def main():
    ap = argparse.ArgumentParser(
        description="Parse a bench.<JOBID>.out log into bench_results.csv + tables + plots."
    )
    ap.add_argument("logfile", help="Path to bench.<JOBID>.out from sbatch bench.slurm")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip matplotlib figure generation")
    args = ap.parse_args()

    rows = parse_log(args.logfile)
    if not rows:
        print(f"ERROR: no [bench] / time= pairs found in {args.logfile}", file=sys.stderr)
        sys.exit(2)

    write_csv(rows)
    g = summarize(rows)
    if not args.no_plots:
        make_plots(g)


if __name__ == "__main__":
    main()
