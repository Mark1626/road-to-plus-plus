#include <benchmark/benchmark.h>
#include "sandpile.hh"

static void BM_SandpileStabilization_1(benchmark::State& state) {
  for (auto _ : state) {
    Fractal::Sandpile sandpile(1 << 4);
    sandpile.computeIdentity();
  }
}
BENCHMARK(BM_SandpileStabilization_1);

static void BM_SandpileStabilization_2(benchmark::State& state) {
  for (auto _ : state) {
    Fractal::Sandpile sandpile(1 << 8);
    sandpile.computeIdentity();
  }
}
BENCHMARK(BM_SandpileStabilization_2);

BENCHMARK_MAIN();
