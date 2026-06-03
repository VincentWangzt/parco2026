#!/usr/bin/env python3
"""Run each binary 11 times for case L (N=1024, T=200), parse the
"compute time: X ms" line, report the median in ms.
Also runs the MPI variants with -np 4 and -np 16."""

import re, statistics, subprocess, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

env = os.environ.copy()
env["PATH"] = "/usr/lib64/openmpi/bin:" + env.get("PATH", "")
env["LD_LIBRARY_PATH"] = "/usr/lib64/openmpi/lib:" + env.get("LD_LIBRARY_PATH", "")
env["OMPI_MCA_pml"] = "ob1"
env["OMPI_MCA_btl"] = "self,vader"

N, T = 1024, 200
RUNS = 11
TIME_RE = re.compile(r"compute time:\s*([\d.eE+-]+)\s*ms")


def time_runs(label, cmd):
    times = []
    for i in range(RUNS):
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        m = TIME_RE.search(out.stdout)
        if not m:
            print(f"[{label}] run {i}: no time line; stdout=\n{out.stdout}\nstderr=\n{out.stderr}")
            sys.exit(1)
        times.append(float(m.group(1)))
    median = statistics.median(times)
    print(f"[{label}] runs={times}")
    print(f"[{label}] median = {median:.3f} ms")
    return median


def main():
    results = {}
    results["serial"]                = time_runs("serial",        ["./serial", str(N), str(T)])
    results["mpi_np4"]               = time_runs("mpi_np4",       ["mpirun", "--allow-run-as-root", "-np", "4",  "./parallel_mpi", str(N), str(T)])
    results["mpi_np16"]              = time_runs("mpi_np16",      ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "16", "./parallel_mpi", str(N), str(T)])
    results["cuda"]                  = time_runs("cuda",          ["./parallel_cuda", str(N), str(T)])
    results["mpi_cuda_np2"]          = time_runs("mpi_cuda_np2",  ["mpirun", "--allow-run-as-root", "-x", "LD_LIBRARY_PATH", "-np", "2", "./parallel_mpi_cuda", str(N), str(T)])
    results["mpi_cuda_np4"]          = time_runs("mpi_cuda_np4",  ["mpirun", "--allow-run-as-root", "-x", "LD_LIBRARY_PATH", "-np", "4", "./parallel_mpi_cuda", str(N), str(T)])

    base = results["serial"]
    print()
    print("=" * 64)
    print(f"Case L (N={N}, T={T}) — median over {RUNS} runs")
    print("=" * 64)
    print(f"{'Variant':<22}{'Time (ms)':>14}{'Speedup':>14}")
    for name, t in results.items():
        sp = base / t if t > 0 else float("inf")
        print(f"{name:<22}{t:>14.3f}{sp:>14.2f}x")


if __name__ == "__main__":
    main()
