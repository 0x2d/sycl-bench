#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Gramschmidt12;
class Gramschmidt3;

void init_array(DATA_TYPE* A, size_t size) {
	const auto M = size;
	const auto N = size;

	for(size_t i = 0; i < M; i++) {
		for(size_t j = 0; j < N; j++) {
			A[i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
		}
	}
}

void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q, size_t size) {
	const auto M = size;
	const auto N = size;

	for(size_t k = 0; k < N; k++) {
		DATA_TYPE nrm = 0;
		for(size_t i = 0; i < M; i++) {
			nrm += A[i * N + k] * A[i * N + k];
		}

		R[k * N + k] = sqrt(nrm);
		for(size_t i = 0; i < M; i++) {
			Q[i * N + k] = A[i * N + k] / R[k * N + k];
		}

		for(size_t j = k + 1; j < N; j++) {
			R[k * N + j] = 0;
			for(size_t i = 0; i < M; i++) {
				R[k * N + j] += Q[i * N + k] * A[i * N + j];
			}
			for(size_t i = 0; i < M; i++) {
				A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
			}
		}
	}
}

class Polybench_Gramschmidt {
  public:
	Polybench_Gramschmidt(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		R.resize(size * size);
		Q.resize(size * size);

		init_array(A.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		R_buffer.initialize(args.device_queue, R.data(), cl::sycl::range<2>(size, size));
		Q_buffer.initialize(args.device_queue, Q.data(), cl::sycl::range<2>(size, size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Gram-Schmidt is sequential in k (each k depends on the prior modified-A
		// columns), so k stays a host loop. The optimization is fusing the
		// baseline's Gram1 (norm) and Gram2 (normalize) kernels into one
		// parallel-reduction kernel -- the [[fdtd2d-opt-kernel-fusion]] lever.
		//
		// The baseline Gram1 is `parallel_for(range<2>(1,1), ...)` -- a SINGLE
		// work-item serially summing N products to get the column norm. That is the
		// worst case (N serial FMAs on one thread). The fused kernel below does the
		// norm as a real parallel tree reduction across one work-group of `local`
		// threads striding over i, writes R[k,k]=sqrt(nrm), barriers, then the same
		// threads write Q[i,k]=A[i,k]/R[k,k] in the same launch. One kernel (not
		// two), and the norm is log(local)-depth instead of N-depth.
		//
		// Gram3 (the dominant O(N^2)-per-k rank-1 update of A and R) is left at
		// the baseline: its two i-loops cannot fuse (R[k,j] needs the full i-sum
		// before any A update), and tiling its Q-column reuse would need a complex
		// nd_range rewrite of a sequential algorithm for a modest, risky win.
		const size_t N = size;
		const size_t local = args.local_size > 0 ? args.local_size : 256;

		for(size_t k = 0; k < N; k++) {
			// Fused Gram1 (norm) + Gram2 (normalize): one work-group of `local`
			// threads striding over the i-axis. R is read_write (write R[{k,k}]
			// then read it back for the divide); Q is write; A is read.
			events.push_back(args.device_queue.submit([&](handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read>(cgh);
				auto R = R_buffer.get_access<access::mode::read_write>(cgh);
				auto Q = Q_buffer.get_access<access::mode::write>(cgh);
				accessor<DATA_TYPE, 1, access::mode::read_write, access::target::local> scratch{local, cgh};

				cgh.parallel_for<Gramschmidt12>(nd_range<1>{local, local}, [=, k_ = k, N_ = N](nd_item<1> item) {
					const size_t lid = item.get_local_id(0);

					// Partial norm from this thread's strided i-slice.
					DATA_TYPE acc = DATA_TYPE(0);
					for(size_t i = lid; i < N_; i += local) {
						acc += A[{i, k_}] * A[{i, k_}];
					}
					scratch[lid] = acc;
					item.barrier(access::fence_space::local_space);

					// Tree reduction -> full norm in scratch[0].
					for(size_t s = local / 2; s > 0; s >>= 1) {
						if(lid < s) scratch[lid] += scratch[lid + s];
						item.barrier(access::fence_space::local_space);
					}

					if(lid == 0) R[{k_, k_}] = cl::sycl::sqrt(scratch[0]);
					item.barrier(access::fence_space::local_space);

					// Normalize: Q[i,k] = A[i,k] / R[k,k].
					const DATA_TYPE rkk = R[{k_, k_}];
					for(size_t i = lid; i < N_; i += local) {
						Q[{i, k_}] = A[{i, k_}] / rkk;
					}
				});
			}));

			// Gram3: rank-1 update of R[k,j] and A[i,j] for j > k (baseline).
			events.push_back(args.device_queue.submit([&](handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read_write>(cgh);
				auto R = R_buffer.get_access<access::mode::write>(cgh);
				auto Q = Q_buffer.get_access<access::mode::read>(cgh);

				cgh.parallel_for<Gramschmidt3>(range<2>(N, 1), [=, M_ = N, N_ = N, k_ = k](item<2> item) {
					const auto j = item[0];

					if(j <= k_ || j >= N_) return;

					R[item] = 0;
					for(size_t i = 0; i < M_; i++) {
						R[item] += Q[{i, k_}] * A[{i, j}];
					}

					for(size_t i = 0; i < M_; i++) {
						A[{i, j}] -= Q[{i, k_}] * R[item];
					}
				});
			}));
		}
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> A_cpu(size * size);
		std::vector<DATA_TYPE> R_cpu(size * size);
		std::vector<DATA_TYPE> Q_cpu(size * size);

		// Trigger writeback
		A_buffer.reset();

		init_array(A_cpu.data(), size);

		gramschmidt(A_cpu.data(), R_cpu.data(), Q_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(A_cpu[i * size + j], A[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Gramschmidt_Opt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> R;
	std::vector<DATA_TYPE> Q;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> R_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> Q_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Gramschmidt>();
	return 0;
}
