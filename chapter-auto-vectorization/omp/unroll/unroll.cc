#include <benchmark/benchmark.h>
#include <cstdlib>

void prod(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int N) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
          C[j*N + i] = A[j*N + i] * B[j*N + i];
      }
  }
}

void prod_unroll(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int N) {
  #pragma omp simd collapse(2)
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
          C[j*N + i] = A[j*N + i] * B[j*N + i];
      }
  }
}

void BM_prod(benchmark::State& state) {
  int N = state.range(0);

  float *A = (float*) malloc(sizeof(float) * N * N);
  float *B = (float*) malloc(sizeof(float) * N * N);
  float *C = (float*) malloc(sizeof(float) * N * N);

  int limit = 1000;

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      A[j*N + i] = (float)std::rand() / (float)(RAND_MAX / limit);
      B[j*N + i] = (float)std::rand() / (float)(RAND_MAX / limit);
      C[j*N + i] = 0;
    }
  }

  for (auto _ : state) {
    prod(A, B, C, N);
    benchmark::DoNotOptimize(C);
  }

  delete [] A;
  delete [] B;
  delete [] C;
}

void BM_prod_unroll(benchmark::State& state) {
  int N = state.range(0);

  float *A = (float*) malloc(sizeof(float) * N * N);
  float *B = (float*) malloc(sizeof(float) * N * N);
  float *C = (float*) malloc(sizeof(float) * N * N);

  int limit = 1000;

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      A[j*N + i] = rand() % limit;
      B[j*N + i] = rand() % limit;
      C[j*N + i] = 0;
    }
  }

  for (auto _ : state) {
    prod_unroll(A, B, C, N);
    benchmark::DoNotOptimize(C);
  }

  delete [] A;
  delete [] B;
  delete [] C;
}

const int st = 1<<8;
const int en = 1<<12;

BENCHMARK(BM_prod)->Range(st, en)->RangeMultiplier(2)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_prod_unroll)->Range(st, en)->RangeMultiplier(2)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
