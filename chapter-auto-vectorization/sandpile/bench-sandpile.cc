#include <cstdint>
#include <string>

#include "sandpile.cc"
#include "sandpile-sse.cc"
#include "sandpile-auto.cc"

#define SIZE 9

#include <benchmark/benchmark.h>


static void BM_Sandpile(benchmark::State &state) {
  for (auto _ : state) {
    size_t pixels = 1 << SIZE;
    Fractal::Sandpile sandpile(pixels);
    sandpile.computeIdentity();
  }
}
BENCHMARK(BM_Sandpile);

static void BM_Sandpile_SSE(benchmark::State &state) {
  for (auto _ : state) {
    size_t pixels = 1 << SIZE;
    FractalSSE::Sandpile sandpile(pixels);
    sandpile.computeIdentity();
  }
}
BENCHMARK(BM_Sandpile_SSE);

static void BM_Sandpile_AutoVec(benchmark::State &state) {
  for (auto _ : state) {
    FractalAutoVec::Sandpile sandpile;
    sandpile.computeIdentity();
  }
}
BENCHMARK(BM_Sandpile_AutoVec);

BENCHMARK_MAIN();
