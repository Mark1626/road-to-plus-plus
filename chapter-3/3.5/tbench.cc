#include "TDigest.hh"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>
#include <omp.h>
#include <vector>

extern int omp_get_thread_num(void);

const int LIMIT_START = 1 << 20;
const int LIMIT_MULTIPLIER = 1 << 3;
const int LIMIT_END = 1 << 24;
const int COMP_START = 1 << 7;
const int COMP_END = 1 << 7;
const int THREADS_START = 12;
const int THREADS_END = 16;

std::vector<double> deserialize(int N, std::string filename = "data.dat") {
  std::vector<double> restore_values;
  std::ifstream ifs(filename);

  if (ifs.is_open()) {
    boost::archive::text_iarchive ia(ifs);

    for (int i = 0; i < N; i++) {
      double value;
      ia >> value;
      restore_values.push_back(value);
    }
  } else {
    throw std::runtime_error("Unable to open file");
  }

  return restore_values;
}

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
  const int N = state.range(0);

  std::vector<double> values = deserialize(N);

  for (auto _ : state) {
    std::sort(values.begin(), values.end());
    auto median = quantile(0.5, values);
    benchmark::DoNotOptimize(median);
  }
}

static void Benchmark_Nth_Elem_Quantile(benchmark::State &state) {
  const int N = state.range(0);

  std::vector<double> values = deserialize(N);

  for (auto _ : state) {
    auto m = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), m, values.end());
    auto median = values[values.size() / 2];
    benchmark::DoNotOptimize(median);
  }
}

static void Benchmark_Nth_Elem_Quantile_MADFM(benchmark::State &state) {
  const int N = state.range(0);

  std::vector<double> values = deserialize(N);

  for (auto _ : state) {
    auto mid = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), mid, values.end());
    auto median = values[values.size() / 2];

    std::vector<double> deviation(values);
    for (size_t i = 0; i < deviation.size(); i++)
      deviation[i] -= median;

    auto deviation_mid = deviation.begin() + deviation.size() / 2;

    std::nth_element(deviation.begin(), deviation_mid, deviation.end());
    auto madfm = deviation[deviation.size() / 2];

    benchmark::DoNotOptimize(median);
    benchmark::DoNotOptimize(madfm);
  }
}

static void Benchmark_Tdigest_Quantile(benchmark::State &state) {
  const int N = state.range(0);
  const double compression = state.range(1);

  TDigest digest(compression);

  std::vector<double> values = deserialize(N);

  for (auto _ : state) {
    for (int i = 0; i < N; i++) {
      digest.add(values[i]);
    }

    auto median = digest.quantile(0.5);
    benchmark::DoNotOptimize(median);
  }
}

static void Benchmark_Parallel_Tdigest_Quantile(benchmark::State &state) {
  const int N = state.range(0);
  const double compression = state.range(1);

  TDigest digest(compression);

  std::vector<double> values = deserialize(N);

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
    // digest.compress();

    auto median = digest.quantile(0.5);

    auto digest_median = digest.quantile(0.5);
    TDigest dfm_digest(compression);

    for (auto centroid : digest.processed) {
      double deviation = digest_median - centroid.mean();
      deviation = deviation < 0 ? -deviation : deviation;
      dfm_digest.add(deviation);
    }

    auto digest_madfm = dfm_digest.quantile(0.5);

    benchmark::DoNotOptimize(median);
    benchmark::DoNotOptimize(digest_madfm);
  }
}


BENCHMARK(Benchmark_Sort_Quantile)
    ->RangeMultiplier(LIMIT_MULTIPLIER)
    ->Range(LIMIT_START, LIMIT_END)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(Benchmark_Tdigest_Quantile)
    ->Ranges({{LIMIT_START, LIMIT_END}, {COMP_START, COMP_END}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(Benchmark_Parallel_Tdigest_Quantile)
    ->Ranges({{LIMIT_START, LIMIT_END},
              {COMP_START, COMP_END},
              {THREADS_START, THREADS_END}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(Benchmark_Nth_Elem_Quantile)
    ->RangeMultiplier(LIMIT_MULTIPLIER)
    ->Range(LIMIT_START, LIMIT_END)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(Benchmark_Nth_Elem_Quantile_MADFM)
    ->RangeMultiplier(LIMIT_MULTIPLIER)
    ->Range(LIMIT_START, LIMIT_END)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
