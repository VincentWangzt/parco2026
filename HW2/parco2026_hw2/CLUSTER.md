# 集群运行流程 (Cluster Pipeline)

Single SLURM entrypoint (`bench.slurm`) covers everything: correctness
(cases S, M) + case L performance + strong-scaling sweep.

| File | Role |
| :--- | :--- |
| `parallel.cpp`, `serial.cpp`, `serial_timed.cpp` | sources |
| `Makefile`           | `mpiicc -O3` build |
| `bench.slurm`        | the only SLURM job |
| `parse_results.py`   | `bench.<JOBID>.out` → CSV / tables / plots |
| `verify.py`          | (provided) checksum |

## Commands to run on the cluster

```bash
ssh cluster
mkdir -p ~/parco_hw2 && cd ~/parco_hw2
# (copy parallel.cpp serial.cpp serial_timed.cpp Makefile bench.slurm
#  parse_results.py verify.py here)

sbatch bench.slurm        # produces bench.<JOBID>.out
squeue -u $USER           # wait for it
```

`bench.slurm` runs three phases in order:

1. **Phase 0** — correctness: `./serial 32` vs `mpirun -np 16 ./parallel
   32` (bit-for-bit `diff` + `verify.py`); `mpirun -np 16 ./parallel
   1000` (the non-divisible `MPI_Gatherv` case) + `verify.py`. Untagged,
   so the parser skips these.
2. **Phase 1** — case L: 5 reps each of `serial_timed` and `parallel
   -np 16` for every `N ∈ NS`.
3. **Phase 2** — strong scaling: 5 reps of `parallel -np p` for every
   `(N, p)` with `N ∈ NS`, `p ∈ PS`. The `p=1` baseline is the parallel
   build, not `serial`, so the speedup measures parallelism only.

Defaults: `NS="8192 16384"`, `PS="1 2 4 8 16"`, `REPS=5`. Override via
`sbatch --export=ALL,NS=...,PS=...,REPS=... bench.slurm`.

## Locally, after the job finishes

```bash
python3 parse_results.py bench.<JOBID>.out
```

Writes `bench_results.csv`, prints the case-L speedup table and the
strong-scaling table (paste straight into `report.md`), and regenerates
`figs/scaling_speedup.png` + `figs/scaling_efficiency.png`. Pass
`--no-plots` if matplotlib isn't installed.

> **Why `serial_timed`?** Same algorithm as `serial.cpp` but wrapped in
> `MPI_Init` / `MPI_Wtime` so the serial baseline uses the *exact* same
> clock as the parallel run. The assignment forbids `clock()` /
> `chrono`. `bench.slurm` invokes it as `mpirun -np 1 ./serial_timed N`.

## Submission

Required: `parallel.cpp`, `report.md` (with the two figures from
`figs/`), `bench.<JOBID>.out`. Optional: `bench_results.csv`.
