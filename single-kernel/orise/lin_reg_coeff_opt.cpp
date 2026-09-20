#include "common.h"
#include <iostream>

//using namespace cl::sycl;
namespace s = cl::sycl;

template <typename T> class CoeffCombineKernel;
template <typename T> class CoeffFinalKernel;

// Optimized twin of single-kernel/lin_reg_coeff.cpp (same fixture, same
// output: coeff_b1/coeff_b0 and the same verify()).
//
// The baseline run() executes, on the device, 14 kernel launches and 4
// full-buffer host synchronizations per measured run:
//   - vec_product x2 (elementwise products into `output`),
//   - reduce() x4 (ss_xy, ss_xx, mean_x's sum, mean_y's sum), each a
//     multi-pass tree reduction (3 passes at 2^20/256) whose final step is a
//     host accessor that maps the WHOLE input buffer back to the host and
//     returns element [0] — i.e. 4 full-buffer D2H copies (4 MB each at 2^20)
//     plus a host stall each.
//
// The opt computes all four sums in ONE fused kernel pass plus one small
// finalize kernel, then reads 32 bytes back to the host:
//
//   * Kernel 1 (combine): one work-group per WG*EPT-element block. Each
//     thread loads in1/in2 for its strided slice once and accumulates four
//     partials per element: in1*in2 (ss_xy), in1*in1 (ss_xx), sum in1
//     (mean_x), sum in2 (mean_y) — oneload-4-uses, so the two input streams
//     are read once instead of being re-read by 2 product kernels and 4
//     reduce passes. The four per-thread partials are tree-reduced over the
//     work-group in a single packed local accessor (4*WG, region s at
//     [s*WG, s*WG+WG)) — halving stride loop, one barrier per level, all
//     four regions advanced per level. lid 0 stores the group's four partial
//     sums into partial_buf[s*ngroups + grpid].
//   * Kernel 2 (finalize): one 256-thread work-group tree-reduces the four
//     ngroups-wide rows of partial_buf and writes the four grand totals to a
//     32-byte results buffer.
//   * Host reads the 32 bytes (one D2H + one sync), then derives the
//     coefficients with the baseline's exact expression order (host T
//     arithmetic, same as verify()).
//
// FP semantics: the fixture data is 1.0/2.0 only, so every product and every
// partial sum (any tree order) is exactly representable in float (sums reach
// at most 2^21 < 2^24) — the device sums are bit-identical to verify()'s
// sequential host sums, and the derived coefficients (computed on the host
// from those sums with the baseline's expressions) are bit-identical too,
// so the 1e-5 tolerance comparison is exact.
//
// Requires a power-of-two --local (harness default 256), same requirement
// the baseline's tree reduction imposes.
template <typename T>
class LinearRegressionCoeffBenchOpt
{
protected:
    std::vector<T> input1;
    std::vector<T> input2;
    std::vector<T> output;

    T coeff_b1;
    T coeff_b0;

    // Only needed for verification (mirrors the baseline's *_ver copies)
    std::vector<T> input1ver;
    std::vector<T> input2ver;
    BenchmarkArgs args;

    PrefetchedBuffer<T, 1> input1_buf;
    PrefetchedBuffer<T, 1> input2_buf;
    PrefetchedBuffer<T, 1> output_buf;

    static constexpr size_t EPT = 8;    // elements per thread, kernel 1
    static constexpr size_t WB = 256;   // finalize work-group width (power of 2)

public:
  LinearRegressionCoeffBenchOpt(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
    // host memory allocation and initialization (identical to baseline)
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    output.resize(args.problem_size, 0);

    input1ver.resize(args.problem_size);
    input2ver.resize(args.problem_size);

    for (size_t i = 0; i < args.problem_size; i++) {
       input1ver[i] = input1[i] = 1.0;
       input2ver[i] = input2[i] = 2.0;
    }

    input1_buf.initialize(args.device_queue, input1.data(), s::range<1>(args.problem_size));
    input2_buf.initialize(args.device_queue, input2.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, output.data(), s::range<1>(args.problem_size));
  }

  void run(std::vector<cl::sycl::event>& events) {
    const size_t N = args.problem_size;

    // Tree reductions require a power-of-two group width.
    size_t WG = args.local_size;
    if (WG == 0 || (WG & (WG - 1))) WG = 256;
    const size_t block = WG * EPT;
    const size_t ngroups = (N + block - 1) / block;

    // Per-group partial sums, four rows: [s*ngroups, s*ngroups+ngroups).
    // s = 0: ss_xy partials, 1: ss_xx, 2: sum_x, 3: sum_y.
    s::buffer<T, 1> partial_buf{s::range<1>(4 * ngroups)};
    s::buffer<T, 1> results_buf{s::range<1>(4)};

    // Kernel 1 — fused multi-sum. One work-group per `block`-element slice.
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.template get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.template get_access<s::access::mode::read>(cgh);
      auto partial = partial_buf.template get_access<s::access::mode::discard_write>(cgh);

      // One packed scratch: region s at [s*WG, s*WG+WG).
      s::accessor<T, 1, s::access::mode::read_write, s::access::target::local>
        scratch{s::range<1>(4 * WG), cgh};

      cgh.parallel_for<CoeffCombineKernel<T>>(
        s::nd_range<1>{ngroups * WG, WG},
        [=, N_ = N, WG_ = WG, EPT_ = EPT, block_ = block, ngroups_ = ngroups]
        (cl::sycl::nd_item<1> item) {
          const size_t lid = item.get_local_linear_id();
          const size_t grpid = item.get_group_linear_id();
          const size_t base = grpid * block_;

          T acc_xy = 0, acc_xx = 0, acc_x = 0, acc_y = 0;
          for (size_t k = 0; k < EPT_; ++k) {
            const size_t idx = base + k * WG_ + lid;
            if (idx < N_) {
              const T v1 = in1[idx];
              const T v2 = in2[idx];
              acc_xy += v1 * v2;
              acc_xx += v1 * v1;
              acc_x += v1;
              acc_y += v2;
            }
          }
          scratch[0 * WG_ + lid] = acc_xy;
          scratch[1 * WG_ + lid] = acc_xx;
          scratch[2 * WG_ + lid] = acc_x;
          scratch[3 * WG_ + lid] = acc_y;
          item.barrier(s::access::fence_space::local_space);

          for (size_t stride = WG_ / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
              scratch[0 * WG_ + lid] += scratch[0 * WG_ + lid + stride];
              scratch[1 * WG_ + lid] += scratch[1 * WG_ + lid + stride];
              scratch[2 * WG_ + lid] += scratch[2 * WG_ + lid + stride];
              scratch[3 * WG_ + lid] += scratch[3 * WG_ + lid + stride];
            }
            item.barrier(s::access::fence_space::local_space);
          }

          if (lid == 0) {
            partial[0 * ngroups_ + grpid] = scratch[0];
            partial[1 * ngroups_ + grpid] = scratch[WG_];
            partial[2 * ngroups_ + grpid] = scratch[2 * WG_];
            partial[3 * ngroups_ + grpid] = scratch[3 * WG_];
          }
        });
    }));

    // Kernel 2 — finalize the four ngroups-wide partial rows.
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto partial = partial_buf.template get_access<s::access::mode::read>(cgh);
      auto results = results_buf.template get_access<s::access::mode::discard_write>(cgh);

      s::accessor<T, 1, s::access::mode::read_write, s::access::target::local>
        scratch{s::range<1>(4 * WB), cgh};

      cgh.parallel_for<CoeffFinalKernel<T>>(
        s::nd_range<1>{WB, WB},
        [=, ngroups_ = ngroups](cl::sycl::nd_item<1> item) {
          const size_t lid = item.get_local_linear_id();

          T acc_xy = 0, acc_xx = 0, acc_x = 0, acc_y = 0;
          for (size_t g = lid; g < ngroups_; g += WB) {
            acc_xy += partial[0 * ngroups_ + g];
            acc_xx += partial[1 * ngroups_ + g];
            acc_x  += partial[2 * ngroups_ + g];
            acc_y  += partial[3 * ngroups_ + g];
          }
          scratch[0 * WB + lid] = acc_xy;
          scratch[1 * WB + lid] = acc_xx;
          scratch[2 * WB + lid] = acc_x;
          scratch[3 * WB + lid] = acc_y;
          item.barrier(s::access::fence_space::local_space);

          for (size_t stride = WB / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
              scratch[0 * WB + lid] += scratch[0 * WB + lid + stride];
              scratch[1 * WB + lid] += scratch[1 * WB + lid + stride];
              scratch[2 * WB + lid] += scratch[2 * WB + lid + stride];
              scratch[3 * WB + lid] += scratch[3 * WB + lid + stride];
            }
            item.barrier(s::access::fence_space::local_space);
          }

          if (lid == 0) {
            results[0] = scratch[0];
            results[1] = scratch[WB];
            results[2] = scratch[2 * WB];
            results[3] = scratch[3 * WB];
          }
        });
    }));

    // Single small D2H + sync (32 bytes). All remaining math is host-side,
    // with the baseline's exact expression order.
    auto results = results_buf.template get_access<s::access::mode::read>();
    const T sum_xy = results[0];
    const T sum_xx = results[1];
    T mean_x = results[2] / N;
    T mean_y = results[3] / N;

    T ss_xy = sum_xy - mean_x * mean_y;
    T ss_xx = sum_xx - mean_x * mean_x;

    coeff_b1 = ss_xy / ss_xx;
    coeff_b0 = mean_y - coeff_b1 * mean_x;
  }

  bool verify(VerificationSetting &ver) {
     bool pass = true;

    T sum_of_vec1 = 0;
    T sum_of_vec2 = 0;
    for (size_t i = 0; i < args.problem_size; i++) {
      sum_of_vec1 += input1ver[i];
      sum_of_vec2 += input2ver[i];
    }

    T mean_x = sum_of_vec1/args.problem_size;
    T mean_y = sum_of_vec2/args.problem_size;

    T ss_xy = 0;
    T ss_xx = 0;
    for (size_t i = 0; i < args.problem_size; i++) {
      ss_xy += input1ver[i]*input2ver[i];
      ss_xx += input1ver[i]*input1ver[i];
    }

    ss_xy = ss_xy - mean_x*mean_y;
    ss_xx = ss_xx - mean_x*mean_x;

    T expected_coeff_b1 = ss_xy/ss_xx;
    T expected_coeff_b0 = mean_y - expected_coeff_b1*mean_x;

    const T tolerance = 0.00001;
    if ((fabs(expected_coeff_b0 - coeff_b0) > tolerance) || (fabs(expected_coeff_b1 - coeff_b1) > tolerance))
      pass = false;

    return pass;
  }

  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "LinearRegressionCoeff_Opt_";
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  if(app.shouldRunNDRangeKernels()){
    app.run<LinearRegressionCoeffBenchOpt<float>>();
    if(app.deviceSupportsFP64())
      app.run<LinearRegressionCoeffBenchOpt<double>>();
  }
  return 0;
}
