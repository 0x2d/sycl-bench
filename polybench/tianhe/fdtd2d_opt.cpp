#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = double;

class Fdtd2d12; // fused ey+ex update (reads hz, writes ex/ey)
class Fdtd2d3;  // hz update (reads ex/ey, writes hz)

constexpr auto TMAX = 500;

void init_arrays(DATA_TYPE* fict, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, size_t size) {
	const auto NX = size;
	const auto NY = size;

	for(size_t i = 0; i < TMAX; i++) {
		fict[i] = (DATA_TYPE)i;
	}

	for(size_t i = 0; i < NX; i++) {
		for(size_t j = 0; j < NY; j++) {
			ex[i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
			ey[i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
			hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
		}
	}
}

void runFdtd(DATA_TYPE* fict, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, size_t size) {
	const auto NX = size;
	const auto NY = size;

	for(size_t t = 0; t < TMAX; t++) {
		for(size_t j = 0; j < NY; j++) {
			ey[0 * NY + j] = fict[t];
		}

		for(size_t i = 1; i < NX; i++) {
			for(size_t j = 0; j < NY; j++) {
				ey[i * NY + j] = ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
			}
		}

		for(size_t i = 0; i < NX; i++) {
			for(size_t j = 1; j < NY; j++) {
				ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] - 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
			}
		}

		for(size_t i = 0; i < NX; i++) {
			for(size_t j = 0; j < NY; j++) {
				hz[i * NY + j] = hz[i * NY + j] - 0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
			}
		}
	}
}

class Polybench_Fdtd2d {
  public:
	Polybench_Fdtd2d(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		fict.resize(TMAX);
		ex.resize(size * (size + 1));
		ey.resize((size + 1) * size);
		hz.resize(size * size);

		init_arrays(fict.data(), ex.data(), ey.data(), hz.data(), size);

		fict_buffer.initialize(args.device_queue, fict.data(), cl::sycl::range<1>(TMAX));
		ex_buffer.initialize(args.device_queue, ex.data(), cl::sycl::range<2>(size, size + 1));
		ey_buffer.initialize(args.device_queue, ey.data(), cl::sycl::range<2>(size + 1, size));
		hz_buffer.initialize(args.device_queue, hz.data(), cl::sycl::range<2>(size, size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Optimized FDTD-2D (MT3K-DSP / tianhe). Two source-level wins, both DSP-safe
		// (no access::target::local, no barriers, no nd_range -- the [[mocl-dsp-constraints]]
		// prohibitions):
		//
		//   1. Fuse the ey update (Fdtd2d1) and ex update (Fdtd2d2) into one kernel.
		//      They read the same hz field and touch disjoint outputs (ey vs ex), so
		//      there is no data dependency between them -- the baseline only runs them
		//      as separate kernels because it is a direct CUDA port. Fusing drops one
		//      kernel launch per timestep (500 launches total) and reads hz[i,j] once
		//      into a named register (hz_ij), reusing it for both updates. The
		//      baseline reads hz[item] twice (once per kernel); under -cl-opt-disable
		//      those are two separate global loads, so the register hoist is a real
		//      traffic win on the DSP (and a no-op where the compiler CSEs it).
		//   2. The hz update (Fdtd2d3) stays its own kernel -- it depends on the
		//      freshly-written ex/ey, so it cannot be fused without a global sync.
		//      Its reads (ex[{i,j+1}], ex[item], ey[{i+1,j}], ey[item]) are all
		//      distinct addresses, so there is no CSE/register-reuse win to chase; it
		//      is left at the baseline access pattern.
		//
		// A shared-memory tiled stencil is deliberately NOT used: on the DSP the
		// barrier/sync cost outweighs the ~2x reuse a 2-point stencil offers, and the
		// local-memory window is too small to hold a useful tile anyway.
		for(size_t t = 0; t < TMAX; t++) {
			events.push_back(args.device_queue.submit([&](handler& cgh) {
				auto fict = fict_buffer.get_access<access::mode::read>(cgh);
				auto ex = ex_buffer.get_access<access::mode::read_write>(cgh);
				auto ey = ey_buffer.get_access<access::mode::read_write>(cgh);
				auto hz = hz_buffer.get_access<access::mode::read>(cgh);

				cgh.parallel_for<Fdtd2d12>(range<2>(size, size), [=, NX_ = size, NY_ = size, t_ = t](item<2> item) {
					const auto i = item[0];
					const auto j = item[1];

					const auto hz_ij = hz[item];

					// ey[i,j] = fict[t] for i==0, else ey[i,j] - 0.5*(hz[i,j] - hz[i-1,j])
					if(i == 0) {
						ey[item] = fict[t_];
					} else {
						ey[item] = ey[item] - 0.5 * (hz_ij - hz[{(i - 1), j}]);
					}

					// ex[i,j] = ex[i,j] - 0.5*(hz[i,j] - hz[i,j-1]) for j>0
					if(j > 0) {
						ex[item] = ex[item] - 0.5 * (hz_ij - hz[{i, (j - 1)}]);
					}
				});
			}));

			events.push_back(args.device_queue.submit([&](handler& cgh) {
				auto ex = ex_buffer.get_access<access::mode::read>(cgh);
				auto ey = ey_buffer.get_access<access::mode::read>(cgh);
				auto hz = hz_buffer.get_access<access::mode::read_write>(cgh);

				cgh.parallel_for<Fdtd2d3>(hz_buffer.get_range(), [=](item<2> item) {
					const auto i = item[0];
					const auto j = item[1];

					hz[item] = hz[item] - 0.7 * (ex[{i, (j + 1)}] - ex[item] + ey[{(i + 1), j}] - ey[item]);
				});
			}));
		}
	}

	bool verify(VerificationSetting&) {
		// Yes, this threshold is used by polybench/CUDA/fdtd2d. Numbers in this
		// benchmark can get pretty large and regular floats don't provide enough
		// precision. This verification may fail on some problem sizes.
		constexpr auto ERROR_THRESHOLD = 10.05;

		std::vector<DATA_TYPE> fict_cpu(TMAX);
		std::vector<DATA_TYPE> ex_cpu(size * (size + 1));
		std::vector<DATA_TYPE> ey_cpu((size + 1) * size);
		std::vector<DATA_TYPE> hz_cpu(size * size);

		// Trigger writebacks
		hz_buffer.reset();

		init_arrays(fict_cpu.data(), ex_cpu.data(), ey_cpu.data(), hz_cpu.data(), size);

		runFdtd(fict_cpu.data(), ex_cpu.data(), ey_cpu.data(), hz_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(hz_cpu[i * size + j], hz[i * size + j]);
				if(diff > ERROR_THRESHOLD) {
					printf("%ld %ld: %f %f %f\n", i, j, hz_cpu[i * size + j], hz[i * size + j], diff);
					return false;
				}
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Fdtd2d_Opt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> fict;
	std::vector<DATA_TYPE> ex;
	std::vector<DATA_TYPE> ey;
	std::vector<DATA_TYPE> hz;

	PrefetchedBuffer<DATA_TYPE, 1> fict_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> ex_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> ey_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> hz_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
        if(app.deviceSupportsFP64())
          app.run<Polybench_Fdtd2d>();
        return 0;
}
