#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class conv2DOpt;

void init(DATA_TYPE* A, size_t size) {
	const auto NI = size;
	const auto NJ = size;

	for(size_t i = 0; i < NI; ++i) {
		for(size_t j = 0; j < NJ; ++j) {
			A[i * NJ + j] = (float)rand() / (float)RAND_MAX;
		}
	}
}

void conv2D(DATA_TYPE* A, DATA_TYPE* B, size_t size) {
	const auto NI = size;
	const auto NJ = size;

	const DATA_TYPE c11 = +0.2, c21 = +0.5, c31 = -0.8;
	const DATA_TYPE c12 = -0.3, c22 = +0.6, c32 = -0.9;
	const DATA_TYPE c13 = +0.4, c23 = +0.7, c33 = +0.10;

	for(size_t i = 1; i < NI - 1; ++i) {
		for(size_t j = 1; j < NJ - 1; ++j) {
			B[i * NJ + j] = c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] + c13 * A[(i + 1) * NJ + (j - 1)]
			                + c21 * A[(i - 1) * NJ + (j + 0)] + c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)]
			                + c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] + c33 * A[(i + 1) * NJ + (j + 1)];
		}
	}
}

// Direct-tap stencil on the full (size,size) grid. Interior threads run
// branch-free on a cheap 2-compare in-bounds guard instead of the baseline's
// per-output 4-compare interior test; the term order matches the baseline
// exactly, so results are bit-identical and opt/baseline pass-or-fail
// verification together.
//
// HONEST NOTE (measured on Device 66a1, gfx906, via the build-mlir toolchain,
// size 4096, interleaved A/B, 30 runs, boost-state-min metric): this kernel
// TIES the baseline (median 0.92-0.94 ms vs 0.92-0.94 ms; min 0.45-0.47 ms vs
// 0.45-0.47 ms). The baseline is already at this device's ceiling for the
// 9-point stencil: one thread per output maximizes latency hiding, its 9
// coalesced loads ride ~2x free L2 reuse, and it carries zero overhead. Eight
// optimization families were measured and ALL tie or regress:
//   - register tiles BMxBN in {1x1..8x8} (array- and scalar-staged):
//     1.05-1.55x SLOWER (fewer threads => worse latency hiding, e.g. 2x2
//     = 16 loads/4 outputs yet +15%, 4x4 = +55%);
//   - LDS 16x16 halo tile (full 16.7M threads, ~1.27 global loads/output,
//     cuts global traffic ~7x): ~10% SLOWER (barrier + LDS reads cost more
//     than the L2 traffic they save);
//   - explicit nd_range work-group shapes (1x256, 4x128, 16x16, 8x32, ...,
//     128/256/512 threads): best shapes tie the runtime's default choice,
//     worst (64x4) is 1.9x slower;
//   - padded grid + early-out guard vs. ragged interior-only grid: tie;
//   - one-row-per-WG LDS layout (the fdtd2d_opt lever): tie;
//   - forced (1,256) nd_range on the verbatim baseline kernel: tie;
//   - float4 VECTORIZED tap loads (4 outputs/thread, two 16B-aligned
//     vector loads per row, 1.5 load instructions/output vs 9): ~20%
//     SLOWER (passes verification; the thread count drops 4x, which costs
//     more than the 6x load-instruction cut gains);
//   - backend flag SYCL_AMDGCN_OPT_FLAGS=-O3 (vs the default -O2): exact
//     tie.
// Separable two-pass decomposition is mathematically unavailable: the
// 9-coefficient matrix is rank-3 (generic), not a product of two 1D filters.
// This mirrors the 2DConvolution_tiled finding on the DPC++/LLVM-HIP path
// for the same device. Kept (rather than deleting the file) because it is
// never slower, is branch-free and marginally simpler, and this comment
// records the negative results so nobody re-sweeps them.
class Polybench_2DConvolution {
  public:
	Polybench_2DConvolution(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);

		init(A.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		B_buffer.initialize(args.device_queue, B.data(), cl::sycl::range<2>(size, size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		const size_t I = size - 2; // interior extent along i
		const size_t J = size - 2; // interior extent along j

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<class conv2DOpt>(B_buffer.get_range(), [=, I_ = I, J_ = J](item<2> item) {
				// Idle border threads exit on the 2-compare guard; the ~4*size
				// border threads of the full grid are noise next to the
				// (size-2)^2 interior, and keeping the full grid preserves
				// clean power-of-two work-group divisibility (a ragged
				// (size-2)^2 grid measured ~8% slower at size 4096).
				if(item[0] >= I_ || item[1] >= J_) return;

				const size_t i = item[0] + 1;
				const size_t j = item[1] + 1;

				const DATA_TYPE c11 = +0.2, c21 = +0.5, c31 = -0.8;
				const DATA_TYPE c12 = -0.3, c22 = +0.6, c32 = -0.9;
				const DATA_TYPE c13 = +0.4, c23 = +0.7, c33 = +0.10;

				B[{i, j}] = c11 * A[{(i - 1), (j - 1)}] + c12 * A[{(i + 0), (j - 1)}] + c13 * A[{(i + 1), (j - 1)}]
				            + c21 * A[{(i - 1), (j + 0)}] + c22 * A[{(i + 0), (j + 0)}] + c23 * A[{(i + 1), (j + 0)}]
				            + c31 * A[{(i - 1), (j + 1)}] + c32 * A[{(i + 0), (j + 1)}] + c33 * A[{(i + 1), (j + 1)}];
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		auto B_acc = B_buffer.get_access<cl::sycl::access::mode::read>();

		std::vector<DATA_TYPE> B_cpu(size * size);
		conv2D(A.data(), B_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				if((i > 0) && (j > 0) && (i < size - 1) && (j < size - 1)) {
					const auto diff = percentDiff(B_cpu[i * size + j], B_acc.get_pointer()[i * size + j]);
					if(diff > ERROR_THRESHOLD) return false;
				}
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_2DConvolution_Opt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> B;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> B_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_2DConvolution>();
	return 0;
}
