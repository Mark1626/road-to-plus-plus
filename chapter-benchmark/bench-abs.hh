#include <cmath>
#include <cstdlib>
#include <xmmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <benchmark/benchmark.h>

static constexpr int RAND_HALF = RAND_MAX / 2;

/*                        Bitwise                             */

int abs1(int a) {
   int s = a >> 31;
   a ^= s;
   a -= s;
   return a;
}

void arr_abs_bitwise(int *__restrict__ a, int *__restrict__ b, int N) {
    for (int i = 0; i < N; i++) {
        b[i] = abs1(a[i]);
    }
}

static void BM_abs_Bitwise(benchmark::State &state) {
  int N = state.range(0);
  int *a = new int[N];
  int *b = new int[N];
  for (int i = 0; i < N; i++) {
    a[i] = rand() - RAND_HALF;
  }

  for (auto _ : state) {
    arr_abs_bitwise(a, b, N);
    benchmark::DoNotOptimize(b);
  }
  delete[](a);
  delete[](b);
}
BENCHMARK(BM_abs_Bitwise)->RangeMultiplier(4)->Range(1<<8, 1<<16)->Complexity();


/*                        Auto Vec                             */

void arr_abs_auto_vec(int * __restrict__ a, int *__restrict__ b, int N) {
    for (int i = 0; i < N; i++) {
        b[i] = abs(a[i]);
    }
}

static void BM_abs_Auto_Vec(benchmark::State &state) {
  int N = state.range(0);
  int *a = new int[N];
  int *b = new int[N];
  for (int i = 0; i < N; i++) {
    a[i] = rand() - RAND_HALF;
  }

  for (auto _ : state) {
    arr_abs_auto_vec(a, b, N);
    benchmark::DoNotOptimize(b);
  }
  delete[](a);
  delete[](b);
}
BENCHMARK(BM_abs_Auto_Vec)->RangeMultiplier(4)->Range(1<<8, 1<<16)->Complexity();

/*                        Manual  SSE                        */

void arr_abs_manual_sse(int *__restrict__ a, int *__restrict__ b, int N) {
    for (int i = 0; i < N/4; i+=4) {
        __m128i av = _mm_loadu_si128((__m128i_u *)a + i);
        __m128i bv = _mm_abs_epi32(av);
        _mm_storeu_si128((__m128i_u *)(b + i), bv);
    }
}

static void BM_abs_Manual_SSE(benchmark::State &state) {
  int N = state.range(0);
  int *a = new int[N];
  int *b = new int[N];
  for (int i = 0; i < N; i++) {
    a[i] = rand() - RAND_HALF;
  }

  for (auto _ : state) {
    arr_abs_manual_sse(a, b, N);
    benchmark::DoNotOptimize(b);
  }
  delete[](a);
  delete[](b);
}
BENCHMARK(BM_abs_Manual_SSE)->RangeMultiplier(4)->Range(1<<8, 1<<16)->Complexity();

/*                        Manual  SSE                        */

void arr_abs_manual_avx(int *__restrict__ a, int *__restrict__ b, int N) {
    for (int i = 0; i < N/8; i+=8) {
        __m256i av = _mm256_loadu_si256((__m256i_u *)a + i);
        __m256i bv = _mm256_abs_epi32(av);
        _mm256_storeu_si256((__m256i_u *)(b + i), bv);
    }
}

static void BM_abs_Manual_AVX(benchmark::State &state) {
  int N = state.range(0);
  int *a = new int[N];
  int *b = new int[N];
  for (int i = 0; i < N; i++) {
    a[i] = rand() - RAND_HALF;
  }

  for (auto _ : state) {
    arr_abs_manual_avx(a, b, N);
    benchmark::DoNotOptimize(b);
  }
  delete[](a);
  delete[](b);
}
BENCHMARK(BM_abs_Manual_AVX)->RangeMultiplier(4)->Range(1<<8, 1<<16)->Complexity();
