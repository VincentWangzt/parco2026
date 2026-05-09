/*
 * Foundations of Parallel Computing II, Spring 2026.
 * Instructor: Chao Yang @ Peking University.
 * OpenMP histogram: unified multi-strategy implementation.
 *
 * Usage: ./histogram_omp <strategy> <input> <output> [runs=5] [warmup=2]
 * Strategies: serial, atomic, critical, private, padded, reduction
 *
 * Timing output (stderr):
 *   TIMING,<strategy>,<threads>,<N>,<M>,<mean_sec>,<std_sec>
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <omp.h>

using namespace std;

// ═══════════════════════════════════════════════════════════════════════════════
// Data and I/O (same contract as serial.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

namespace utils {
    int N;           // number of data points
    int M;           // number of bins
    double min_val;  // minimum value of the range
    double max_val;  // maximum value of the range

    vector<double> data;
    vector<int>    hist;

    // Utility: check if a filename ends with a given suffix
    static bool ends_with(const string& s, const string& suffix) {
        if (s.size() < suffix.size()) return false;
        return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    void abort_with_error_message(const string& msg) {
        cerr << msg << endl;
        abort();
    }

    void read_inputs(const string& filename) {
        if (ends_with(filename, ".bin")) {
            // Binary format: [N:i32][M:i32][min_val:f64][max_val:f64][data: N×f64]
            FILE* f = fopen(filename.c_str(), "rb");
            if (!f)
                abort_with_error_message("ERROR: Unable to open input file.");
            fread(&N, sizeof(int), 1, f);
            fread(&M, sizeof(int), 1, f);
            fread(&min_val, sizeof(double), 1, f);
            fread(&max_val, sizeof(double), 1, f);
            data.resize(N);
            fread(data.data(), sizeof(double), N, f);
            fclose(f);
        } else {
            // Text format: "N M min_val max_val\n" followed by N doubles
            ifstream f(filename);
            if (!f.is_open())
                abort_with_error_message("ERROR: Unable to open input file.");
            f >> N >> M >> min_val >> max_val;
            data.resize(N);
            for (int i = 0; i < N; ++i)
                f >> data[i];
            f.close();
        }

        hist.resize(M, 0);
    }

    void write_outputs(const string& filename) {
        if (ends_with(filename, ".bin")) {
            // Binary format: [hist: M × int32]
            FILE* f = fopen(filename.c_str(), "wb");
            if (!f)
                abort_with_error_message("ERROR: Unable to open output file.");
            fwrite(hist.data(), sizeof(int), M, f);
            fclose(f);
        } else {
            // Text format: one integer per line
            ofstream f(filename);
            if (!f.is_open())
                abort_with_error_message("ERROR: Unable to open output file.");
            for (int i = 0; i < M; ++i)
                f << hist[i] << "\n";
            f.close();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Strategy Implementations
// ═══════════════════════════════════════════════════════════════════════════════

// Strategy 1: Serial baseline (single-threaded reference)
void hist_serial() {
    double bin_width = (utils::max_val - utils::min_val) / utils::M;
    for (int i = 0; i < utils::N; ++i) {
        int bin = (int)((utils::data[i] - utils::min_val) / bin_width);
        if (bin < 0)  bin = 0;
        if (bin >= utils::M) bin = utils::M - 1;
        utils::hist[bin]++;
    }
}

// Strategy 2: Atomic — #pragma omp atomic on hist[bin]++
void hist_atomic() {
    double bin_width = (utils::max_val - utils::min_val) / utils::M;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < utils::N; ++i) {
        int bin = (int)((utils::data[i] - utils::min_val) / bin_width);
        if (bin < 0)  bin = 0;
        if (bin >= utils::M) bin = utils::M - 1;
        #pragma omp atomic
        utils::hist[bin]++;
    }
}

// Strategy 3: Critical section — worst performance, for comparison
void hist_critical() {
    double bin_width = (utils::max_val - utils::min_val) / utils::M;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < utils::N; ++i) {
        int bin = (int)((utils::data[i] - utils::min_val) / bin_width);
        if (bin < 0)  bin = 0;
        if (bin >= utils::M) bin = utils::M - 1;
        #pragma omp critical
        { utils::hist[bin]++; }
    }
}

// Strategy 4: Thread-private arrays with manual reduction (no padding)
// Arrays are contiguous — may exhibit false sharing at thread boundaries
void hist_private() {
    double bin_width = (utils::max_val - utils::min_val) / utils::M;
    int nthreads = omp_get_max_threads();
    int M = utils::M;

    // Contiguous allocation: nthreads × M (no padding)
    vector<int> local_hists(nthreads * M, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int* my_hist = &local_hists[tid * M];

        #pragma omp for schedule(static)
        for (int i = 0; i < utils::N; ++i) {
            int bin = (int)((utils::data[i] - utils::min_val) / bin_width);
            if (bin < 0)  bin = 0;
            if (bin >= M) bin = M - 1;
            my_hist[bin]++;
        }
    }

    // Manual reduction: merge all thread-local histograms
    for (int t = 0; t < nthreads; ++t)
        for (int b = 0; b < M; ++b)
            utils::hist[b] += local_hists[t * M + b];
}

// Strategy 5: Thread-private arrays with cache-line padding + manual reduction
// Padding eliminates false sharing between adjacent thread arrays
void hist_padded() {
    const int CACHE_LINE = 64;
    const int INTS_PER_LINE = CACHE_LINE / sizeof(int);  // 16

    double bin_width = (utils::max_val - utils::min_val) / utils::M;
    int nthreads = omp_get_max_threads();
    int M = utils::M;

    // Pad each thread's array to cache-line boundary, plus one extra line gap
    int padded_M = ((M + INTS_PER_LINE - 1) / INTS_PER_LINE) * INTS_PER_LINE;
    int stride = padded_M + INTS_PER_LINE;  // extra cache line between threads

    vector<int> local_hists(nthreads * stride, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int* my_hist = &local_hists[tid * stride];

        #pragma omp for schedule(static)
        for (int i = 0; i < utils::N; ++i) {
            int bin = (int)((utils::data[i] - utils::min_val) / bin_width);
            if (bin < 0)  bin = 0;
            if (bin >= M) bin = M - 1;
            my_hist[bin]++;
        }
    }

    // Manual reduction
    for (int t = 0; t < nthreads; ++t)
        for (int b = 0; b < M; ++b)
            utils::hist[b] += local_hists[t * stride + b];
}

// Strategy 6: OpenMP 4.5+ array section reduction (compiler-managed)
/*
void hist_reduction() {
    double bin_width = (utils::max_val - utils::min_val) / utils::M;
    int M = utils::M;
    int* hist_arr = utils::hist.data();

    #pragma omp parallel for schedule(static) reduction(+:hist_arr[:M])
    for (int i = 0; i < utils::N; ++i) {
        int bin = (int)((utils::data[i] - utils::min_val) / bin_width);
        if (bin < 0)  bin = 0;
        if (bin >= M) bin = M - 1;
        hist_arr[bin]++;
    }
}
*/

// ═══════════════════════════════════════════════════════════════════════════════
// Strategy Registry
// ═══════════════════════════════════════════════════════════════════════════════

typedef void (*StrategyFunc)();

struct StrategyEntry {
    const char* name;
    StrategyFunc func;
};

static StrategyEntry strategies[] = {
    {"serial",    hist_serial},
    {"atomic",    hist_atomic},
    {"critical",  hist_critical},
    {"private",   hist_private},
    {"padded",    hist_padded},
//    {"reduction", hist_reduction},
};
static const int NUM_STRATEGIES = sizeof(strategies) / sizeof(strategies[0]);

// ═══════════════════════════════════════════════════════════════════════════════
// Main: Timing Harness
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: ./histogram_omp <strategy> <input> <output> [runs=5] [warmup=2]" << endl;
        cerr << "Strategies:";
        for (int i = 0; i < NUM_STRATEGIES; ++i)
            cerr << " " << strategies[i].name;
        cerr << endl;
        return 1;
    }

    string strategy_name = argv[1];
    int num_runs = (argc > 4) ? atoi(argv[4]) : 5;
    int warmup   = (argc > 5) ? atoi(argv[5]) : 2;

    // Find strategy
    StrategyFunc func = nullptr;
    for (int i = 0; i < NUM_STRATEGIES; ++i) {
        if (strategy_name == strategies[i].name) {
            func = strategies[i].func;
            break;
        }
    }
    if (!func) {
        cerr << "ERROR: Unknown strategy '" << strategy_name << "'" << endl;
        return 1;
    }

    utils::read_inputs(argv[2]);

    // Warmup runs
    for (int r = 0; r < warmup; ++r) {
        fill(utils::hist.begin(), utils::hist.end(), 0);
        func();
    }

    // Timed runs
    vector<double> times(num_runs);
    for (int r = 0; r < num_runs; ++r) {
        fill(utils::hist.begin(), utils::hist.end(), 0);
        double t0 = omp_get_wtime();
        func();
        double t1 = omp_get_wtime();
        times[r] = t1 - t0;
    }

    // Compute mean and std
    double mean = accumulate(times.begin(), times.end(), 0.0) / num_runs;
    double sq_sum = 0.0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = sqrt(sq_sum / num_runs);

    // Output timing to stderr (parseable)
    cerr << "TIMING," << strategy_name << ","
         << omp_get_max_threads() << ","
         << utils::N << "," << utils::M << ","
         << mean << "," << stddev << endl;

    // Write last run's histogram to output file
    utils::write_outputs(argv[3]);

    return 0;
}
