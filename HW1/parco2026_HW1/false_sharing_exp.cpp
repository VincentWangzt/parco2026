/*
 * Foundations of Parallel Computing II, Spring 2026.
 * Instructor: Chao Yang @ Peking University.
 * False Sharing Experiment: measure impact of cache-line padding.
 *
 * Uses thread-private histogram strategy with configurable padding.
 * M is read from the input file; padding is specified via CLI.
 *
 * Usage: ./false_sharing_exp <input> <padding_bytes> <num_threads> [runs=5] [warmup=2]
 *
 * Output to stderr:
 *   FALSESHARE,<M>,<padding_bytes>,<threads>,<N>,<mean_sec>,<std_sec>
 * Output to stdout: histogram (M lines) for correctness verification
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
#include <omp.h>

using namespace std;

static int N, M;
static double min_val, max_val;
static vector<double> data;
static vector<int> hist;

void read_inputs(const string& filename) {
    ifstream f(filename);
    if (!f.is_open()) { cerr << "ERROR: Cannot open " << filename << endl; abort(); }
    f >> N >> M >> min_val >> max_val;
    data.resize(N);
    for (int i = 0; i < N; ++i)
        f >> data[i];
    f.close();
    hist.resize(M, 0);
}

/*
 * Thread-private histogram with configurable padding (in bytes).
 *
 * Memory layout per thread (stride in ints):
 *   [M ints of histogram] [padding_ints of dead space]
 *
 * When padding_bytes=0 and M*4 is not a multiple of 64, the last cache line
 * of thread T's array overlaps with the first cache line of thread T+1's array,
 * causing false sharing (cache-line invalidation on every write).
 */
void compute_histogram_with_padding(int padding_bytes, int nthreads) {
    int padding_ints = (padding_bytes + sizeof(int) - 1) / sizeof(int);  // ceil
    int stride = M + padding_ints;  // ints per thread

    vector<int> local_hists(nthreads * stride, 0);
    double bin_width = (max_val - min_val) / M;

    omp_set_num_threads(nthreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int* my_hist = &local_hists[tid * stride];

        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            int bin = (int)((data[i] - min_val) / bin_width);
            if (bin < 0)  bin = 0;
            if (bin >= M) bin = M - 1;
            my_hist[bin]++;
        }
    }

    // Manual reduction
    for (int t = 0; t < nthreads; ++t)
        for (int b = 0; b < M; ++b)
            hist[b] += local_hists[t * stride + b];
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: ./false_sharing_exp <input> <padding_bytes> <num_threads> [runs=5] [warmup=2]" << endl;
        return 1;
    }

    string input_file = argv[1];
    int padding_bytes = atoi(argv[2]);
    int nthreads      = atoi(argv[3]);
    int num_runs      = (argc > 4) ? atoi(argv[4]) : 5;
    int warmup        = (argc > 5) ? atoi(argv[5]) : 2;

    read_inputs(input_file);

    // Warmup
    for (int r = 0; r < warmup; ++r) {
        fill(hist.begin(), hist.end(), 0);
        compute_histogram_with_padding(padding_bytes, nthreads);
    }

    // Timed runs
    vector<double> times(num_runs);
    for (int r = 0; r < num_runs; ++r) {
        fill(hist.begin(), hist.end(), 0);
        double t0 = omp_get_wtime();
        compute_histogram_with_padding(padding_bytes, nthreads);
        double t1 = omp_get_wtime();
        times[r] = t1 - t0;
    }

    // Compute mean and std
    double mean = accumulate(times.begin(), times.end(), 0.0) / num_runs;
    double sq_sum = 0.0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = sqrt(sq_sum / num_runs);

    // Output timing
    cerr << "FALSESHARE," << M << "," << padding_bytes << ","
         << nthreads << "," << N << "," << mean << "," << stddev << endl;

    // Write histogram to stdout (for correctness check via redirection)
    for (int i = 0; i < M; ++i)
        cout << hist[i] << "\n";

    return 0;
}
