#include <cstdint>
#include <string>

#include "sandpile.cc"
#include "sandpile-sse.cc"
#include "sandpile-auto.cc"

#include "Config.h"

#include <benchmark/benchmark.h>

static void BM_Sandpile(benchmark::State &state) {
  size_t pixels = 1 << SIZE;
  Fractal::Sandpile sandpile(pixels);
  for (auto _ : state) {
    sandpile.computeIdentity();
  }
}
BENCHMARK(BM_Sandpile)->Unit(benchmark::kMillisecond);

static void BM_Sandpile_SSE(benchmark::State &state) {
  size_t pixels = 1 << SIZE;
  FractalSSE::Sandpile sandpile(pixels);
  for (auto _ : state) {
    sandpile.computeIdentity();
  }
}
BENCHMARK(BM_Sandpile_SSE)->Unit(benchmark::kMillisecond);

static void BM_Sandpile_AutoVec(benchmark::State &state) {
  FractalAutoVec::Sandpile sandpile;
  for (auto _ : state) {
    sandpile.computeIdentity();
  }
}
BENCHMARK(BM_Sandpile_AutoVec)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
