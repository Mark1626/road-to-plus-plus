#include <benchmark/benchmark.h>
#include <cstdlib>

float work(float *__restrict__ a, float *__restrict__ b, int n) {
  int i;
  float tmp, sum;
  sum = 0.0;
#pragma omp simd private(tmp) reduction(+ : sum)
  for (i = 0; i < n; i++) {
    tmp = a[i] + b[i];
    sum += tmp;
  }
  return sum;
}

float work_simd(float *__restrict__ a, float *__restrict__ b, int n) {
  int i;
  float tmp, sum;
  sum = 0.0;
#pragma omp simd
  for (i = 0; i < n; i++) {
    tmp = a[i] + b[i];
    sum += tmp;
  }
  return sum;
}

float work_simd_reduce(float *__restrict__ a, float *__restrict__ b, int n) {
  int i;
  float tmp, sum;
  sum = 0.0;
#pragma omp simd private(tmp) reduction(+ : sum)
  for (i = 0; i < n; i++) {
    tmp = a[i] + b[i];
    sum += tmp;
  }
  return sum;
}

inline void BM_work_template(benchmark::State &state,
                      float fn(float *__restrict__ a, float *__restrict__ b,
                               int n)) {
  int N = state.range(0);

  float *A = (float *)malloc(sizeof(float) * N);
  float *B = (float *)malloc(sizeof(float) * N);

  int limit = 1000;

  for (int j = 0; j < N; j++) {
    A[j] = (float)std::rand() / (float)(RAND_MAX / limit);
    B[j] = (float)std::rand() / (float)(RAND_MAX / limit);
  }

  for (auto _ : state) {
    float res = fn(A, B, N);
    benchmark::DoNotOptimize(res);
  }

  delete[] A;
  delete[] B;
}

void BM_work(benchmark::State &state) { BM_work_template(state, work); }

void BM_work_simd(benchmark::State &state) {
  BM_work_template(state, work_simd);
}

void BM_work_simd_reduction(benchmark::State &state) {
  BM_work_template(state, work_simd_reduce);
}

const int st = 1 << 10;
const int en = 1 << 24;

BENCHMARK(BM_work)->Range(st, en)->RangeMultiplier(2);
BENCHMARK(BM_work_simd)->Range(st, en)->RangeMultiplier(2);
BENCHMARK(BM_work_simd_reduction)->Range(st, en)->RangeMultiplier(2);

BENCHMARK_MAIN();
