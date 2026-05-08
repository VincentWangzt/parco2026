/*
 * Foundations of Parallel Computing II, Spring 2026.
 * Instructor: Chao Yang @ Peking University.
 * Optimized parallel histogram — initial version for iterative refinement.
 *
 * Usage: ./histogram_fast <input> <output>
 *
 * Optimizations applied:
 * 1. Thread-private histograms with cache-line padding
 * 2. Aligned memory allocation (posix_memalign)
 * 3. Reciprocal multiply (avoid division in hot loop)
 * 4. schedule(static) for spatial locality
 * 5. Parallel merge phase
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
static const int NUM_RUNS = 5;
static const int WARMUP_RUNS = 2;
// ────────────────────────────────────────────────────────────────────────────

static int N, M;
static double min_val, max_val;
static double* data_arr = nullptr;
static int* hist_arr = nullptr;

// Aligned allocation
template<typename T>
static T* aligned_alloc_array(size_t count) {
    void* ptr = nullptr;
    posix_memalign(&ptr, CACHE_LINE, count * sizeof(T));
    return static_cast<T*>(ptr);
}

static void read_inputs(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", filename); abort(); }

    fscanf(f, "%d %d %lf %lf", &N, &M, &min_val, &max_val);
    data_arr = aligned_alloc_array<double>(N);
    for (int i = 0; i < N; ++i)
        fscanf(f, "%lf", &data_arr[i]);
    fclose(f);

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

    int* local_hists = aligned_alloc_array<int>(nthreads * stride);
    memset(local_hists, 0, nthreads * stride * sizeof(int));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        int* __restrict__ my_hist = local_hists + tid * stride;

        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            int bin = (int)((data_arr[i] - dmin) * inv_bin_width);
            // Clamp to valid range
            if (bin < 0) bin = 0;
            if (bin >= Mlocal) bin = Mlocal - 1;
            my_hist[bin]++;
        }
    }

    // Parallel merge
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < Mlocal; ++b) {
        int sum = 0;
        for (int t = 0; t < nthreads; ++t)
            sum += local_hists[t * stride + b];
        hist_arr[b] = sum;
    }

    free(local_hists);
}

static void write_outputs(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", filename); abort(); }
    for (int i = 0; i < M; ++i)
        fprintf(f, "%d\n", hist_arr[i]);
    fclose(f);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: ./histogram_fast <input> <output>\n");
        return 1;
    }

    read_inputs(argv[1]);

    // Warmup
    for (int r = 0; r < WARMUP_RUNS; ++r) {
        memset(hist_arr, 0, M * sizeof(int));
        compute_histogram();
    }

    // Timed runs
    double times[NUM_RUNS];
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
