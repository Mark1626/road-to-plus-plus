#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <stdint.h>
#include <sys/types.h>

///////////////////// Benchmark Template ////////////////////////////////////

void benchmark_template(benchmark::State &state, void fn(float*, float*, float*, uint32_t, uint32_t)) {
  int N = state.range(0);
  int K = rand() % N;
  
  float *a = new float[N];
  float *b = new float[N];
  float *c = new float[N];

  float range = 3.5f;

  for (int i = 0; i < N; i++) {
    a[i] = (double)std::rand() / (double)(RAND_MAX / range);
    b[i] = (double)std::rand() / (double)(RAND_MAX / range);
    c[i] = 0.0f;
  }

  for (auto _ : state) {
    fn(a, b, c, N, K);
  }

  benchmark::DoNotOptimize(c);

  delete [] a;
  delete [] b;
  delete [] c;
}

inline void fn_accumulate(float *a, float *b,  float *c, uint32_t N, uint32_t K) {
  for (uint32_t i = 0; i < N; i++) {
    c[K] += a[i] * b[i];
  }
}

inline void fn_auto_vec_fix_accumulate(float *a, float *b,  float *c, uint32_t N, uint32_t K) {
  float sum = 0;
  for (uint32_t i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }
  c[K] = sum;
}

inline void fn_omp_simd_accumulate(float *a, float *b,  float *c, uint32_t N, uint32_t K) {
  float sum = 0;

  #pragma omp simd reduction(+:sum)
  for (uint32_t i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }
  c[K] = sum;
}

void BM_Accumulate_Mul(benchmark::State &state) {
  benchmark_template(state, fn_accumulate);
}

void BM_Accumulate_Mul_Auto_Vec(benchmark::State &state) {
  benchmark_template(state, fn_auto_vec_fix_accumulate);
}

void BM_Accumulate_Mul_SIMD(benchmark::State &state) {
  benchmark_template(state, fn_omp_simd_accumulate);
}

uint32_t st = 1 << 18;
uint32_t en = 1 << 26;

// auto range = Ranges({{st, en}})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Accumulate_Mul)->Ranges({{st, en}})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Accumulate_Mul_Auto_Vec)->Ranges({{st, en}})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Accumulate_Mul_SIMD)->Ranges({{st, en}})->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
