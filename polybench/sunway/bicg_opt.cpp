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

		// Optimized BICG: s = A^T * r  and  q = A * p, A in row-major (N x N).
		//
		// The baseline (polybench/bicg.cpp) launches each kernel with
		// parallel_for(range<1>(N)) and an auto-chosen local size (~256). With
		// N=16384 that is only ~64 work-groups — barely one wavefront per CU on
		// this DCU — so the device cannot hide memory latency and BICG runs far
		// below peak bandwidth. The win below is raising occupancy + fixing
		// coalescing via a per-output work-group reduction, NOT the RMW collapse
		// (the compiler already hoists `s[item] +=` / `q[item] +=` into a
		// register, so a plain register-accumulator rewrite measures no gain on
		// its own). This mirrors the atax_opt result, with the kernel roles
		// swapped because BICG's two GEMVs touch the opposite axis of A.
		//
		//   * Bicg2 (q = A*p): baseline maps consecutive work-items to
		//     consecutive ROWS => `A[{i,j}], A[{i+1,j}]` strided by N =>
		//     UNCOALESCED. This is the slow kernel. Fix: one work-group per row
		//     i, WG threads cooperate over j with a stride of WG. Consecutive
		//     local ids map to consecutive j -> A[{i,j}] reads are contiguous in
		//     row-major storage, i.e. coalesced across the wavefront. Partial
		//     sums are reduced in local memory (tree reduction). Thread count
		//     rises from N to N*WG, restoring occupancy. The win lives here: its
		//     reduction axis (j) is the contiguous axis of A.
		//   * Bicg1 (s = A^T*r): baseline maps consecutive work-items to
		//     consecutive COLUMNS => `A[{i,j}], A[{i,j+1}]` adjacent =>
		//     ALREADY coalesced. Its reduction axis is i (the strided axis), so
		//     a per-column work-group would make the A reads strided (one column
		//     = stride N), wrecking coalescing — measured worse in the atax
		//     twin. So Bicg1 keeps the baseline one-thread-per-column mapping
		//     (reads coalesced *across* work-items) and only collapses the
		//     `s[item] +=` RMW into a register accumulator + single store.
		const size_t N = size;
		// Tree reduction requires a power-of-two group width.
		const size_t local = args.local_size > 0 ? args.local_size : 128;

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto r = r_buffer.get_access<access::mode::read>(cgh);
			auto s = s_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Bicg1>(s_buffer.get_range(), [=, N_ = N](item<1> item) {
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
			accessor<DATA_TYPE, 1, access::mode::read_write, access::target::local> scratch{local, cgh};

			cgh.parallel_for<Bicg2>(nd_range<1>{N * local, local}, [=, N_ = N](nd_item<1> item) {
				const size_t lid = item.get_local_id(0);
				const size_t i = item.get_group(0);

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t jj = lid; jj < N_; jj += local) {
					acc += A[{i, jj}] * p[jj];
				}
				scratch[lid] = acc;
				item.barrier(access::fence_space::local_space);

				for(size_t s = local / 2; s > 0; s >>= 1) {
					if(lid < s) scratch[lid] += scratch[lid + s];
					item.barrier(access::fence_space::local_space);
				}

				if(lid == 0) q[i] = scratch[0];
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
