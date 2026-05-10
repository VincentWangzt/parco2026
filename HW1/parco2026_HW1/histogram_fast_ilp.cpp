/*
 * Foundations of Parallel Computing II, Spring 2026.
 * Instructor: Chao Yang @ Peking University.
 * Optimized parallel histogram with dual ILP optimization.
 *
 * Usage: ./histogram_fast <input> <output> [num_runs] [warmup_runs]
 *
 * Optimizations applied:
 * 1. Thread-private histograms with cache-line padding
 * 2. Aligned memory allocation (posix_memalign)
 * 3. Reciprocal multiply (avoid division in hot loop)
 * 4. schedule(static) for spatial locality
 * 5. Parallel merge phase
 * 6. Dual private histograms per thread (ILP) — breaks RAW dependency chain
 *
 * Timing output (stderr):
 *   TIMING,fast,<threads>,<N>,<M>,<mean_sec>,<std_sec>
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>

using namespace std;

// ─── Configuration ──────────────────────────────────────────────────────────
static const int CACHE_LINE = 64;
static const int INTS_PER_LINE = CACHE_LINE / sizeof(int);  // 16
static int NUM_RUNS = 5;
static int WARMUP_RUNS = 2;
static const int NREP = 2;  // dual histograms for ILP
// ────────────────────────────────────────────────────────────────────────────

static int N, M;
static double min_val, max_val;
static double* data_arr = nullptr;
static int* hist_arr = nullptr;

// Utility: check if a filename ends with a given suffix
static bool ends_with(const char* s, const char* suffix) {
    size_t slen = strlen(s), sufflen = strlen(suffix);
    if (slen < sufflen) return false;
    return strcmp(s + slen - sufflen, suffix) == 0;
}

// Aligned allocation
template<typename T>
static T* aligned_alloc_array(size_t count) {
    void* ptr = nullptr;
    posix_memalign(&ptr, CACHE_LINE, count * sizeof(T));
    return static_cast<T*>(ptr);
}

static void read_inputs(const char* filename) {
    if (ends_with(filename, ".bin")) {
        // Binary format: [N:i32][M:i32][min_val:f64][max_val:f64][data: N×f64]
        FILE* f = fopen(filename, "rb");
        if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", filename); abort(); }
        fread(&N, sizeof(int), 1, f);
        fread(&M, sizeof(int), 1, f);
        fread(&min_val, sizeof(double), 1, f);
        fread(&max_val, sizeof(double), 1, f);
        data_arr = aligned_alloc_array<double>(N);
        fread(data_arr, sizeof(double), N, f);
        fclose(f);
    } else {
        // Text format: "N M min_val max_val\n" followed by N doubles
        FILE* f = fopen(filename, "r");
        if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", filename); abort(); }
        fscanf(f, "%d %d %lf %lf", &N, &M, &min_val, &max_val);
        data_arr = aligned_alloc_array<double>(N);
        for (int i = 0; i < N; ++i)
            fscanf(f, "%lf", &data_arr[i]);
        fclose(f);
    }

    hist_arr = aligned_alloc_array<int>(M);
    memset(hist_arr, 0, M * sizeof(int));
}

static void compute_histogram() {
    const int nthreads = omp_get_max_threads();
    const double inv_bin_width = (double)M / (max_val - min_val);  // reciprocal
    const double dmin = min_val;
    const int Mlocal = M;

    // Padded stride: round up M to cache-line boundary + 1 line gap
    const int padded_M = ((Mlocal + INTS_PER_LINE - 1) / INTS_PER_LINE) * INTS_PER_LINE;
    const int stride = padded_M + INTS_PER_LINE;  // +1 cache line between threads

    // Allocate NREP histogram copies per thread for ILP
    int* local_hists = aligned_alloc_array<int>(nthreads * NREP * stride);
    memset(local_hists, 0, nthreads * NREP * stride * sizeof(int));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        // Two histogram copies per thread to break RAW dependency chain
        int* __restrict__ hist0 = local_hists + (tid * NREP + 0) * stride;
        int* __restrict__ hist1 = local_hists + (tid * NREP + 1) * stride;

        // Alternate between hist0 and hist1 on consecutive elements.
        // This breaks the Read-After-Write dependency: the CPU can pipeline
        // the increment to hist1 while hist0's increment is still in flight.
        #pragma omp for schedule(static)
        for (int i = 0; i < N - 1; i += 2) {
            int bin0 = (int)((data_arr[i] - dmin) * inv_bin_width);
            if (bin0 < 0) bin0 = 0;
            if (bin0 >= Mlocal) bin0 = Mlocal - 1;
            hist0[bin0]++;

            int bin1 = (int)((data_arr[i + 1] - dmin) * inv_bin_width);
            if (bin1 < 0) bin1 = 0;
            if (bin1 >= Mlocal) bin1 = Mlocal - 1;
            hist1[bin1]++;
        }

        // Handle the last element if N is odd
        #pragma omp single
        if (N % 2 != 0) {
            int bin = (int)((data_arr[N - 1] - dmin) * inv_bin_width);
            if (bin < 0) bin = 0;
            if (bin >= Mlocal) bin = Mlocal - 1;
            hist0[bin]++;
        }
    }

    // Parallel merge: accumulate all thread-local histograms into hist_arr
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < Mlocal; ++b) {
        int sum = 0;
        for (int t = 0; t < nthreads; ++t)
            for (int r = 0; r < NREP; ++r)
                sum += local_hists[(t * NREP + r) * stride + b];
        hist_arr[b] = sum;
    }

    free(local_hists);
}

static void write_outputs(const char* filename) {
    if (ends_with(filename, ".bin")) {
        // Binary format: [hist: M × int32]
        FILE* f = fopen(filename, "wb");
        if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", filename); abort(); }
        fwrite(hist_arr, sizeof(int), M, f);
        fclose(f);
    } else {
        // Text format: one integer per line
        FILE* f = fopen(filename, "w");
        if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", filename); abort(); }
        for (int i = 0; i < M; ++i)
            fprintf(f, "%d\n", hist_arr[i]);
        fclose(f);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: ./histogram_fast <input> <output> [num_runs] [warmup_runs]\n");
        return 1;
    }

    if (argc >= 4) NUM_RUNS = atoi(argv[3]);
    if (argc >= 5) WARMUP_RUNS = atoi(argv[4]);

    read_inputs(argv[1]);

    // Warmup
    for (int r = 0; r < WARMUP_RUNS; ++r) {
        memset(hist_arr, 0, M * sizeof(int));
        compute_histogram();
    }

    // Timed runs
    vector<double> times(NUM_RUNS);
    for (int r = 0; r < NUM_RUNS; ++r) {
        memset(hist_arr, 0, M * sizeof(int));
        double t0 = omp_get_wtime();
        compute_histogram();
        double t1 = omp_get_wtime();
        times[r] = t1 - t0;
    }

    // Mean and std
    double mean = 0.0;
    for (int r = 0; r < NUM_RUNS; ++r) mean += times[r];
    mean /= NUM_RUNS;

    double sq_sum = 0.0;
    for (int r = 0; r < NUM_RUNS; ++r) sq_sum += (times[r] - mean) * (times[r] - mean);
    double stddev = sqrt(sq_sum / NUM_RUNS);

    fprintf(stderr, "TIMING,fast,%d,%d,%d,%.6f,%.6f\n",
            omp_get_max_threads(), N, M, mean, stddev);

    write_outputs(argv[2]);
    free(data_arr);
    free(hist_arr);
    return 0;
}
