/*
 * Foundations of Parallel Computing II, Spring 2026.
 * Instructor: Chao Yang @ Peking University.
 * This is a serial implementation of histogram computation.
 */

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

namespace utils {
    int N;           // number of data points
    int M;           // number of bins
    double min_val;  // minimum value of the range
    double max_val;  // maximum value of the range

    vector<double> data;
    vector<int>    hist;

    void abort_with_error_message(const string& msg) {
        cerr << msg << endl;
        abort();
    }

    /*
     * The input file format:
     * - Line 1: N M min_val max_val
     * - Following N lines: one floating-point number per line
     *
     * The output file format:
     * - M lines: one integer (bin count) per line
     *
     * Run:
     * $ ./serial <input_file> <output_file>
     */
    void read_inputs(const string& filename) {
        ifstream f(filename);
        if (!f.is_open())
            abort_with_error_message("ERROR: Unable to open input file.");

        f >> N >> M >> min_val >> max_val;
        data.resize(N);
        for (int i = 0; i < N; ++i)
            f >> data[i];
        f.close();

        hist.resize(M, 0);
    }

    void compute_histogram() {
        double bin_width = (max_val - min_val) / M;
        for (int i = 0; i < N; ++i) {
            int bin = (int)((data[i] - min_val) / bin_width);
            if (bin < 0)  bin = 0;
            if (bin >= M) bin = M - 1;
            hist[bin]++;
        }
    }

    void write_outputs(const string& filename) {
        ofstream f(filename);
        if (!f.is_open())
            abort_with_error_message("ERROR: Unable to open output file.");

        for (int i = 0; i < M; ++i)
            f << hist[i] << "\n";
        f.close();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3)
        utils::abort_with_error_message("ERROR: Usage: ./serial <input_file> <output_file>");

    utils::read_inputs(argv[1]);
    utils::compute_histogram();
    utils::write_outputs(argv[2]);

    return 0;
}
