#include "tdigest.cc"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

static double quantile(const double q, const std::vector<double> &values) {
  double q1;
  if (values.size() == 0) {
    q1 = NAN;
  } else if (q == 1 || values.size() == 1) {
    q1 = values[values.size() - 1];
  } else {
    auto index = q * values.size();
    if (index < 0.5) {
      q1 = values[0];
    } else if (values.size() - index < 0.5) {
      q1 = values[values.size() - 1];
    } else {
      index -= 0.5;
      const int intIndex = static_cast<int>(index);
      q1 = values[intIndex + 1] * (index - intIndex) +
           values[intIndex] * (intIndex + 1 - index);
    }
  }
  return q1;
}

static void Benchmark_Sort_Quantile(benchmark::State &state) {
  std::uniform_real_distribution<> reals(0.0, 1.0);
  std::random_device gen;
  const int N = state.range(0);

  std::vector<double> values;
  for (int i = 0; i < N; i++) {
    auto value = reals(gen);
    values.push_back(value);
  }

  for (auto _ : state) {
    std::sort(values.begin(), values.end());
    auto median = quantile(0.5, values);
    benchmark::DoNotOptimize(median);
  }
}

static void Benchmark_Nth_Elem_Quantile(benchmark::State &state) {
  std::uniform_real_distribution<> reals(0.0, 1.0);
  std::random_device gen;
  const int N = state.range(0);

  std::vector<double> values;
  for (int i = 0; i < N; i++) {
    auto value = reals(gen);
    values.push_back(value);
  }

  for (auto _ : state) {
    auto m = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), m, values.end());
    auto median = values[values.size() / 2];
    benchmark::DoNotOptimize(median);
  }
}

static void Benchmark_Tdigest_Quantile(benchmark::State &state) {
  std::uniform_real_distribution<> reals(0.0, 1.0);
  std::random_device gen;
  const int N = state.range(0);
  const double compression = state.range(1);

  TDigest digest(compression);

  std::vector<double> values;
  for (int i = 0; i < N; i++) {
    auto value = reals(gen);
    values.push_back(value);
  }

  for (auto _ : state) {
    for (int i = 0; i < N; i++) {
      digest.add(values[i]);
    }

    auto median = digest.quantile(0.5);
    benchmark::DoNotOptimize(median);
  }
}

static void Benchmark_Parallel_Tdigest_Quantile(benchmark::State &state) {
  std::uniform_real_distribution<> reals(0.0, 1.0);
  std::random_device gen;
  const int N = state.range(0);
  const double compression = state.range(1);

  TDigest digest(compression);

  std::vector<double> values;
  for (int i = 0; i < N; i++) {
    auto value = reals(gen);
    values.push_back(value);
  }

  const int threads = state.range(2);
  int n_sections = threads;
  int section_width = N / n_sections;

  for (auto _ : state) {

#pragma omp parallel for num_threads(threads)
    for (int section = 0; section < n_sections; section++) {
      TDigest local_digest(compression);

      for (int i = 0; i < section_width; i++) {
        int idx = (section * section_width) + i;
        local_digest.add(values[idx]);
      }

#pragma omp critical
      { digest.merge(&local_digest); }
    }
    // digest.add(local_digests.begin(), local_digests.end());
    digest.compress();

    auto median = digest.quantile(0.5);
    benchmark::DoNotOptimize(median);
  }
}

BENCHMARK(Benchmark_Sort_Quantile)
    ->RangeMultiplier(1 << 3)
    ->Range(1 << 10, 1 << 24)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(Benchmark_Tdigest_Quantile)
    ->Ranges({{1 << 10, 1 << 24}, {1 << 7, 1 << 8}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(Benchmark_Parallel_Tdigest_Quantile)
    ->Ranges({{1 << 10, 1 << 24}, {1 << 7, 1 << 8}, {4, 12}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(Benchmark_Nth_Elem_Quantile)
    ->RangeMultiplier(1 << 3)
    ->Range(1 << 10, 1 << 24)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
