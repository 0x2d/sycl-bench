#ifndef SUITE_PARAMETERS_HPP
#define SUITE_PARAMETERS_HPP

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>

// Fixed benchmark parameters as configured by bin/run-suite-orise.
//
// Benchmark executables report the problem-size/local-size result columns
// from these values instead of the --size/--local values they were invoked
// with, so results are always labelled with the suite's canonical
// parameters no matter how the benchmark was called. The benchmark itself
// still runs with the parameters passed on the command line - only the
// reported values are fixed.
//
// This table mirrors default_profile in bin/run-suite-orise:
//   * --local is never overridden per benchmark, so every benchmark runs
//     with local size 256 (create_log_range(256,256)).
//   * --size defaults to 2^10 for benchmarks without an entry in
//     'individual-benchmark-options' (e.g. 2mm, 3mm, gemm, syr2k, syrk and
//     all *_opt variants).
//   * Keys are the executable names (basename of argv[0], without a
//     directory or .exe suffix), copied verbatim from the script. Some keys
//     (e.g. 'segmentatedreduction') match no executable built by this
//     repo and therefore never override anything - exactly as in the
//     script, which keys on the installed file names.
struct SuiteParameters {
  std::size_t problem_size;
  std::size_t local_size;
};

// Returns the fixed parameters reported for the given executable name
// (e.g. 'atax', 'lin_reg_error', 'atax_opt'), or nullptr if the benchmark
// has no fixed parameters: dag_task_throughput_independent/sequential are
// swept over a range of sizes (2^10..2^16) by the suite, so there is no
// single fixed value and their passed-in parameters are reported unchanged.
inline const SuiteParameters* fixedSuiteParameters(const std::string& executable_name)
{
  static const SuiteParameters default_parameters{std::size_t{1} << 10, 256};

  static const std::unordered_set<std::string> no_fixed_parameters = {
    "dag_task_throughput_independent",
    "dag_task_throughput_sequential",
  };

  // --size overrides from run-suite-orise's
  // default_profile['individual-benchmark-options'].
  static const std::unordered_map<std::string, SuiteParameters> fixed_sizes = {
    {"pattern_L2", {std::size_t{1} << 20, 256}},
    {"pattern_arith", {std::size_t{1} << 20, 256}},
    {"pattern_sf", {std::size_t{1} << 20, 256}},
    {"pattern_shared", {std::size_t{1} << 20, 256}},
    {"kmeans", {std::size_t{1} << 20, 256}},
    {"lin_reg_coeff", {std::size_t{1} << 20, 256}},
    {"lin_reg_error", {std::size_t{1} << 16, 256}},
    {"mol_dyn", {std::size_t{1} << 20, 256}},
    {"scalar_prod", {std::size_t{1} << 20, 256}},
    {"vec_add", {std::size_t{1} << 20, 256}},
    {"blocked_transform", {std::size_t{1} << 20, 256}},
    {"reduction", {std::size_t{1} << 20, 256}},
    {"segmentatedreduction", {std::size_t{1} << 20, 256}},
    {"2DConvolution", {std::size_t{1} << 12, 256}},
    {"atax", {std::size_t{1} << 12, 256}},
    {"bicg", {std::size_t{1} << 14, 256}},
    {"gesummv", {std::size_t{1} << 14, 256}},
    {"mvt", {std::size_t{1} << 14, 256}},
    {"usm_accessors_latency", {std::size_t{1} << 20, 256}},
    {"usm_allocation_latency", {std::size_t{1} << 25, 256}},
    {"usm_instr_mix", {std::size_t{1} << 14, 256}},
    {"usm_pinned_overhead", {std::size_t{1} << 20, 256}},
    {"spec_constant_convolution", {std::size_t{1} << 11, 256}},
    {"atomic_reduction", {std::size_t{1} << 20, 256}},
    {"reduce_over_group", {std::size_t{1} << 20, 256}},
    {"kernel_reduction", {std::size_t{1} << 20, 256}},
  };

  if(no_fixed_parameters.count(executable_name) > 0)
    return nullptr;

  auto it = fixed_sizes.find(executable_name);
  if(it != fixed_sizes.end())
    return &it->second;

  // Benchmarks not listed in run-suite-orise run with the default size.
  return &default_parameters;
}

#endif
