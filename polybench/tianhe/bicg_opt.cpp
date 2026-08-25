#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

#ifndef M_PI
#define M_PI 3.14159
#endif

using DATA_TYPE = float;

class Bicg1;
class Bicg2;

void init_array(DATA_TYPE* A, DATA_TYPE* p, DATA_TYPE* r, size_t size) {
	const auto NX = size;
	const auto NY = size;

	for(size_t i = 0; i < NX; i++) {
		r[i] = i * M_PI;

		for(size_t j = 0; j < NY; j++) {
			A[i * NY + j] = ((DATA_TYPE)i * j) / NX;
		}
	}

	for(size_t i = 0; i < NY; i++) {
		p[i] = i * M_PI;
	}
}

void bicg_cpu(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q, size_t size) {
	const auto NX = size;
	const auto NY = size;

	for(size_t i = 0; i < NX; i++) {
		for(size_t j = 0; j < NY; j++) {
			s[j] += r[i] * A[i * NY + j];
			q[i] += A[i * NY + j] * p[j];
		}
	}
}

class Polybench_Bicg {
  public:
	Polybench_Bicg(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		r.resize(size);
		s.resize(size);
		p.resize(size);
		q.resize(size);

		init_array(A.data(), p.data(), r.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		r_buffer.initialize(args.device_queue, r.data(), cl::sycl::range<1>(size));
	  s_buffer.initialize(args.device_queue, s.data(), cl::sycl::range<1>(size));
		p_buffer.initialize(args.device_queue, p.data(), cl::sycl::range<1>(size));
		q_buffer.initialize(args.device_queue, q.data(), cl::sycl::range<1>(size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Optimized BICG (MT3K-DSP / tianhe): s = A^T * r  and  q = A * p, A row-major (N x N).
		//
		// This is the atax_opt lever (the [[mocl-dsp-constraints]] register-accumulator
		// rewrite) applied to both BICG kernels. The baseline (polybench/bicg.cpp)
		// does `s[item] += A*r` / `q[item] += A*p` inside the reduction loop. Under the
		// DSP build's POCL_EXTRA_BUILD_FLAGS="-cl-opt-disable" the compiler does NOT
		// hoist s[item]/q[item] into a register, so the output is touched N times per
		// element -- the main inefficiency on the MT3K DSP. Accumulating in a scalar
		// register and storing once fixes that (the bulk of the win), exactly like the
		// 1x1 register tile in 2mm_opt/3mm_opt and the Atax1/Atax2 kernels.
		//
		//   * No access::target::local, no barriers, no nd_range: the MT3K DSP's
		//     per-work-group on-chip local memory is too small to hold a useful shared
		//     tile and 1D local accessors + barriers hang the MOCL backend (see the
		//     previous bicg_opt work-group tree-reduction). The orise twin raises
		//     occupancy via a per-row work-group reduction in local memory; that path
		//     is dropped here because it is unavailable on the DSP. On CPU/GPU -- where
		//     the device compiler already hoists the `+=` into a register -- the
		//     register-accumulator rewrite is a no-op, so this version is never a
		//     regression there.
		//   * Bicg1 (s = A^T*r): one thread per column j, reduction over i (the strided
		//     axis of A). The per-column mapping keeps A reads coalesced *across*
		//     work-items (adjacent columns j are adjacent in memory). s is written from
		//     zero (discard_write; s starts at 0).
		//   * Bicg2 (q = A*p): one thread per row i, reduction over j (the contiguous
		//     axis of A -> A[{i,j}] reads coalesced). q is written from zero
		//     (discard_write; q starts at 0).

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto r = r_buffer.get_access<access::mode::read>(cgh);
			auto s = s_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<Bicg1>(s_buffer.get_range(), [=, N_ = size](item<1> item) {
				const auto j = item[0];

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t i = 0; i < N_; i++) {
					acc += A[{i, j}] * r[i];
				}
				s[item] = acc;
			});
		}));

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto p = p_buffer.get_access<access::mode::read>(cgh);
			auto q = q_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<Bicg2>(q_buffer.get_range(), [=, N_ = size](item<1> item) {
				const auto i = item[0];

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t j = 0; j < N_; j++) {
					acc += A[{i, j}] * p[j];
				}
				q[item] = acc;
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		// Trigger writebacks
		s_buffer.reset();
		q_buffer.reset();

		std::vector<DATA_TYPE> s_cpu(size);
		std::vector<DATA_TYPE> q_cpu(size);

		bicg_cpu(A.data(), r.data(), s_cpu.data(), p.data(), q_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			auto diff = percentDiff(s_cpu[i], s[i]);
			if(diff > ERROR_THRESHOLD) return false;

			diff = percentDiff(q_cpu[i], q[i]);
			if(diff > ERROR_THRESHOLD) return false;
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Bicg_Opt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> r;
	std::vector<DATA_TYPE> s;
	std::vector<DATA_TYPE> p;
	std::vector<DATA_TYPE> q;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> r_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> s_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> p_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> q_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Bicg>();
	return 0;
}
