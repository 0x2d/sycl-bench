#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class CovarianceMean;
class CovarianceReduce;
class CovarianceCovar;

constexpr DATA_TYPE float_n = 3214212.01;

void init_arrays(DATA_TYPE* data, size_t size) {
	const auto M = size;
	const auto N = size;

	for(size_t i = 0; i < M; i++) {
		for(size_t j = 0; j < N; j++) {
			data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
		}
	}
}

void covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean, size_t size) {
	const auto M = size;
	const auto N = size;

	// Determine mean of column vectors of input data matrix
	for(size_t j = 1; j <= M; j++) {
		mean[j] = 0.0;
		for(size_t i = 1; i <= N; i++) {
			mean[j] += data[i * (M + 1) + j];
		}
		mean[j] /= float_n;
	}

	// Center the column vectors.
	for(size_t i = 1; i <= N; i++) {
		for(size_t j = 1; j <= M; j++) {
			data[i * (M + 1) + j] -= mean[j];
		}
	}

	// Calculate the m * m covariance matrix.
	for(size_t j1 = 1; j1 <= M; j1++) {
		for(size_t j2 = j1; j2 <= M; j2++) {
			symmat[j1 * (M + 1) + j2] = 0.0;
			for(size_t i = 1; i <= N; i++) {
				symmat[j1 * (M + 1) + j2] += data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
			}
			symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
		}
	}
}

class Polybench_Covariance {
public:
	Polybench_Covariance(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		data.resize((size + 1) * (size + 1));
		symmat.resize((size + 1) * (size + 1));
		mean.resize(size + 1);

		init_arrays(data.data(), size);

		data_buffer.initialize(args.device_queue, data.data(), cl::sycl::range<2>(size + 1, size + 1));
		symmat_buffer.initialize(args.device_queue, symmat.data(), cl::sycl::range<2>(size + 1, size + 1));
		mean_buffer.initialize(args.device_queue, mean.data(), cl::sycl::range<1>(size + 1));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Optimized covariance (MT3K-DSP / tianhe). The dominant O(N^3) Covar kernel
		// and the O(N^2) Mean kernel both get the [[mocl-dsp-constraints]]
		// register-accumulator rewrite; Reduce is elementwise (single RMW, not a
		// loop) and is left at baseline.
		//
		//   * Mean: baseline does `mean[item] += data[{i,j}]` per i -- a global RMW
		//     N times per element. Under -cl-opt-disable the compiler does not hoist
		//     mean[item] into a register. Accumulate in a scalar register, divide by
		//     float_n, store once (discard_write; mean starts at 0).
		//   * Reduce: `data[item] -= mean[j]` -- one read + one write per element,
		//     not a loop, so no N-fold RMW to collapse. Left as-is.
		//   * Covar: baseline maps one thread per j1, serially looping j2 then i with
		//     `symmat[{j1,j2}] += data*data` per i (RMW N times per output, plus the
		//     outer j2 loop is serial per thread -> ~N^2 serial FMAs per thread, very
		//     low occupancy). The rewrite maps one thread per output (j1,j2),
		//     accumulates over i in a scalar register, and stores once -- collapsing
		//     the per-i RMW. Symmetry is exploited: only the upper triangle (j2 >= j1)
		//     is computed; the lower triangle is filled by a transposed mirror store.
		//     The diagonal is the variance sum_i D[i,j]^2 (acc), NOT 1.0 (covariance
		//     has no unit-diagonal constraint, unlike correlation).
		//   * No access::target::local, no barriers, no nd_range: the orise twin's
		//     shared-memory tiled Covar hangs/crashes on the DSP (tiny local-memory
		//     window; 1D/2D local accessors + barriers unsupported by MOCL). On
		//     CPU/GPU the device compiler already hoists the `+=` into a register, so
		//     the register-accumulator rewrite is a no-op there -- never a regression.
		constexpr size_t OFFSET = 1; // polybench active submatrix lives at [1..size]

		// Mean of column vectors.
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh);
			auto mean = mean_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<CovarianceMean>(range<1>(size), id<1>(OFFSET), [=, N_ = size](item<1> item) {
				const auto j = item[0];

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t i = 1; i <= N_; i++) {
					acc += data[{i, j}];
				}
				mean[item] = acc / float_n;
			});
		}));

		// Center the column vectors (baseline -- covariance only subtracts the mean,
		// no division by stddev/sqrt(float_n) unlike correlation).
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto mean = mean_buffer.get_access<access::mode::read>(cgh);
			auto data = data_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<CovarianceReduce>(range<2>(size, size), id<2>(OFFSET, OFFSET), [=](item<2> item) {
				const auto j = item[1];
				data[item] -= mean[j];
			});
		}));

		// Covariance matrix: symmat = D^T * D, one thread per output (j1, j2),
		// register accumulator over i. Upper triangle + mirror store.
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh);
			auto symmat = symmat_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<CovarianceCovar>(range<2>(size, size), [=, N_ = size](item<2> item) {
				const size_t lj1 = item[0]; // logical j1 in [0, size)
				const size_t lj2 = item[1]; // logical j2 in [0, size)

				// Lower-triangle cell: filled by the mirror store of the symmetric
				// (j2, j1) thread. Skip to avoid computing it twice.
				if(lj2 < lj1) return;

				const size_t j1 = OFFSET + lj1;
				const size_t j2 = OFFSET + lj2;

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t i = 1; i <= N_; i++) {
					acc += data[{i, j1}] * data[{i, j2}];
				}

				symmat[{j1, j2}] = acc;
				if(lj2 > lj1) {
					symmat[{j2, j1}] = acc; // mirror into the lower triangle
				}
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> data_cpu((size + 1) * (size + 1));
		std::vector<DATA_TYPE> symmat_cpu((size + 1) * (size + 1));
		std::vector<DATA_TYPE> mean_cpu(size + 1);

		// Trigger writeback
		symmat_buffer.reset();

		init_arrays(data_cpu.data(), size);

		covariance(data_cpu.data(), symmat_cpu.data(), mean_cpu.data(), size);

		for(size_t i = 1; i < size + 1; i++) {
			for(size_t j = 1; j < size + 1; j++) {
				const auto diff = percentDiff(symmat_cpu[i * (size + 1) + j], symmat[i * (size + 1) + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Covariance_Opt"; }

private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> data;
	std::vector<DATA_TYPE> symmat;
	std::vector<DATA_TYPE> mean;

	PrefetchedBuffer<DATA_TYPE, 2> data_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> symmat_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> mean_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Covariance>();
	return 0;
}
