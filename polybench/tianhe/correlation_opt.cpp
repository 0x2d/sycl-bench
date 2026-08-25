#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

#define FLOAT_N 3214212.01
#define EPS 0.005

#define sqrt_of_array_cell(x, j) sqrt(x[j])

using DATA_TYPE = float;

class CorrelationMean;
class CorrelationStd;
class CorrelationReduce;
class CorrelationCorr;

void init_arrays(DATA_TYPE* data, size_t size) {
	const auto M = size;
	const auto N = size;

	for(size_t i = 0; i <= M; i++) {
		for(size_t j = 0; j <= N; j++) {
			data[i * N + j] = ((DATA_TYPE)i * j) / (M + 1);
		}
	}
}

void correlation(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat, size_t size) {
	const auto M = size;
	const auto N = size;

	// Determine mean of column vectors of input data matrix
	for(size_t j = 1; j <= M; j++) {
		mean[j] = 0.0;

		for(size_t i = 1; i <= N; i++) {
			mean[j] += data[i * (M + 1) + j];
		}

		mean[j] /= (DATA_TYPE)FLOAT_N;
	}

	// Determine standard deviations of column vectors of data matrix.
	for(size_t j = 1; j <= M; j++) {
		stddev[j] = 0.0;

		for(size_t i = 1; i <= N; i++) {
			stddev[j] += (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
		}

		stddev[j] /= FLOAT_N;
		stddev[j] = sqrt_of_array_cell(stddev, j);
		stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
	}

	// Center and reduce the column vectors.
	for(size_t i = 1; i <= N; i++) {
		for(size_t j = 1; j <= M; j++) {
			data[i * (M + 1) + j] -= mean[j];
			data[i * (M + 1) + j] /= sqrt(FLOAT_N);
			data[i * (M + 1) + j] /= stddev[j];
		}
	}

	// Calculate the m * m correlation matrix.
	for(size_t j1 = 1; j1 <= M - 1; j1++) {
		symmat[j1 * (M + 1) + j1] = 1.0;

		for(size_t j2 = j1 + 1; j2 <= M; j2++) {
			symmat[j1 * (M + 1) + j2] = 0.0;

			for(size_t i = 1; i <= N; i++) {
				symmat[j1 * (M + 1) + j2] += (data[i * (M + 1) + j1] * data[i * (M + 1) + j2]);
			}

			symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
		}
	}

	symmat[M * (M + 1) + M] = 1.0;
}

class Polybench_Correlation {
public:
	Polybench_Correlation(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		data.resize((size + 1) * (size + 1));
		mean.resize(size + 1);
		stddev.resize(size + 1);
		symmat.resize((size + 1) * (size + 1));

		init_arrays(data.data(), size);

		data_buffer.initialize(args.device_queue, data.data(), cl::sycl::range<2>(size + 1, size + 1));
		mean_buffer.initialize(args.device_queue, mean.data(), cl::sycl::range<1>(size + 1));
		stddev_buffer.initialize(args.device_queue, stddev.data(), cl::sycl::range<1>(size + 1));
		symmat_buffer.initialize(args.device_queue, symmat.data(), cl::sycl::range<2>(size + 1, size + 1));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Optimized correlation (MT3K-DSP / tianhe). Mean/Std get the
		// [[mocl-dsp-constraints]] register-accumulator rewrite (baseline does
		// `mean[item] +=` / `stddev[item] +=` per i -- a global RMW N times per
		// element that -cl-opt-disable does not hoist); Reduce and Corr likewise.
		//
		//   * Mean: scalar-register accumulate, divide by FLOAT_N, store once.
		//   * Std: scalar-register accumulate the squared deviations, then divide /
		//     sqrt / EPS-clamp, store once.
		//   * Reduce: `data -= mean; data /= sqrt(FLOAT_N); data /= stddev` -- three
		//     RMWs of data[item] per element. Hoist data[item] into a named register
		//     so the three reads collapse to one (under -cl-opt-disable each RMW is a
		//     separate global reload).
		//   * Corr: baseline maps one thread per j1, serially looping j2 then i with
		//     `symmat[{j1,j2}] += data*data` per i (RMW), plus a separate Correlation5
		//     kernel to set symmat[M,M]=1.0. The rewrite maps one thread per output
		//     (j1,j2), accumulates over i in a scalar register, stores once, exploits
		//     symmetry (upper triangle + mirror), and forces the diagonal to 1.0 (a
		//     variable correlates perfectly with itself) -- subsuming Correlation5.
		//   * No access::target::local, no barriers, no nd_range: the orise twin's
		//     shared-memory tiled Corr hangs/crashes on the DSP. On CPU/GPU the device
		//     compiler already hoists the `+=` into a register, so the register-
		//     accumulator rewrite is a no-op there -- never a regression.
		constexpr size_t OFFSET = 1; // polybench active submatrix lives at [1..size]

		// Mean of column vectors.
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh);
			auto mean = mean_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<CorrelationMean>(range<1>(size), id<1>(OFFSET), [=, N_ = size](item<1> item) {
				const auto j = item[0];

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t i = 1; i <= N_; i++) {
					acc += data[{i, j}];
				}
				mean[item] = acc / ((DATA_TYPE)FLOAT_N);
			});
		}));

		// Standard deviations of column vectors.
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh);
			auto mean = mean_buffer.get_access<access::mode::read>(cgh);
			auto stddev = stddev_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<CorrelationStd>(range<1>(size), id<1>(OFFSET), [=, N_ = size](item<1> item) {
				const auto j = item[0];
				const DATA_TYPE m = mean[item];

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t i = 1; i <= N_; i++) {
					const DATA_TYPE d = data[{i, j}] - m;
					acc += d * d;
				}

				acc /= FLOAT_N;
				acc = cl::sycl::sqrt(acc);
				acc = acc <= EPS ? DATA_TYPE(1.0) : acc;
				stddev[item] = acc;
			});
		}));

		// Center and reduce the column vectors.
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read_write>(cgh);
			auto mean = mean_buffer.get_access<access::mode::read>(cgh);
			auto stddev = stddev_buffer.get_access<access::mode::read>(cgh);

			const DATA_TYPE inv_sqrt_fn = DATA_TYPE(1.0) / cl::sycl::sqrt((DATA_TYPE)FLOAT_N);

			cgh.parallel_for<CorrelationReduce>(range<2>(size, size), id<2>(OFFSET, OFFSET), [=](item<2> item) {
				const auto j = item[1];

				DATA_TYPE d = data[item];
				d -= mean[j];
				d *= inv_sqrt_fn; // /= sqrt(FLOAT_N)
				d /= stddev[j];
				data[item] = d;
			});
		}));

		// Correlation matrix: symmat = D^T * D, one thread per output (j1, j2),
		// register accumulator over i. Diagonal forced to 1.0 (subsumes Correlation5);
		// upper off-diagonal mirrored into the lower triangle.
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh);
			auto symmat = symmat_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<CorrelationCorr>(range<2>(size, size), [=, N_ = size](item<2> item) {
				const size_t lj1 = item[0]; // logical j1 in [0, size)
				const size_t lj2 = item[1]; // logical j2 in [0, size)

				// Lower-triangle cell: filled by the mirror store of the symmetric
				// (j2, j1) thread. Skip to avoid computing it twice.
				if(lj2 < lj1) return;

				const size_t j1 = OFFSET + lj1;
				const size_t j2 = OFFSET + lj2;

				if(lj1 == lj2) {
					// Diagonal: a variable correlates perfectly with itself.
					symmat[{j1, j2}] = DATA_TYPE(1.0);
					return;
				}

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t i = 1; i <= N_; i++) {
					acc += data[{i, j1}] * data[{i, j2}];
				}

				symmat[{j1, j2}] = acc;
				symmat[{j2, j1}] = acc; // mirror into the lower triangle
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> data_cpu((size + 1) * (size + 1));
		std::vector<DATA_TYPE> mean_cpu(size + 1);
		std::vector<DATA_TYPE> stddev_cpu(size + 1);
		std::vector<DATA_TYPE> symmat_cpu((size + 1) * (size + 1));

		// Trigger writeback
		symmat_buffer.reset();

		init_arrays(data_cpu.data(), size);
		correlation(data_cpu.data(), mean_cpu.data(), stddev_cpu.data(), symmat_cpu.data(), size);

		for(size_t i = 1; i < size + 1; i++) {
			for(size_t j = 1; j < size + 1; j++) {
				const auto diff = percentDiff(symmat_cpu[i * (size + 1) + j], symmat[i * (size + 1) + j]);
				if(diff > ERROR_THRESHOLD)
					return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Correlation_Opt"; }

private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> data;
	std::vector<DATA_TYPE> mean;
	std::vector<DATA_TYPE> stddev;
	std::vector<DATA_TYPE> symmat;

	PrefetchedBuffer<DATA_TYPE, 2> data_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> mean_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> stddev_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> symmat_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Correlation>();
	return 0;
}
