#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <xmmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <benchmark/benchmark.h>

static constexpr const int N = 1<<16;
static constexpr int RAND_HALF = RAND_MAX / 2;

/* Max bitwise */

int max_bitwise(int a, int b) {
  int diff = a - b;
  int dsgn = diff >> 31;
  return a - (diff & dsgn);
}

void arr_max_bitwise(int *__restrict__ a, int *__restrict__ b, int *__restrict__ c, int N) {
  for (int i = 0; i < N; i++) {
    c[i] = max_bitwise(a[i], b[i]);
  }
}

static void BM_max_Bitwise(benchmark::State &state) {
  int N = state.range(0);
  int *a = new int[N];
  int *b = new int[N];
  int *c = new int[N];
  for (int i = 0; i < N; i++) {
    a[i] = rand() - RAND_HALF;
    b[i] = rand() - RAND_HALF;
  }

  for (auto _ : state) {
    arr_max_bitwise(a, b, c, N);
    benchmark::DoNotOptimize(c);
  }
  delete[](a);
  delete[](b);
  delete[](c);
}
BENCHMARK(BM_max_Bitwise)->RangeMultiplier(8)->Range(1<<8, 1<<24)->Complexity();


/* Max std::max */

void arr_std_max(int *__restrict__ a, int *__restrict__ b, int *__restrict__ c, int N) {
  for (int i = 0; i < N; i++) {
    c[i] = std::max(a[i], b[i]);
  }
}

static void BM_max_Std_algo(benchmark::State &state) {
  int N = state.range(0);
  int *a = new int[N];
  int *b = new int[N];
  int *c = new int[N];
  for (int i = 0; i < N; i++) {
    a[i] = rand() - RAND_HALF;
    b[i] = rand() - RAND_HALF;
  }

  for (auto _ : state) {
    arr_std_max(a, b, c, N);
    benchmark::DoNotOptimize(c);
  }
  delete[](a);
  delete[](b);
  delete[](c);
}
BENCHMARK(BM_max_Std_algo)->RangeMultiplier(8)->Range(1<<8, 1<<24)->Complexity();

/* Max normal */
void arr_max(int *__restrict__ a, int *__restrict__ b, int *__restrict__ c, int N) {
  for (int i = 0; i < N; i++) {
    c[i] = (a[i] > b[i]) ? a[i] : b[i];
  }
}

static void BM_max_Normal(benchmark::State &state) {
  int N = state.range(0);
  int *a = new int[N];
  int *b = new int[N];
  int *c = new int[N];
  for (int i = 0; i < N; i++) {
    a[i] = rand() - RAND_HALF;
    b[i] = rand() - RAND_HALF;
  }

  for (auto _ : state) {
    arr_max(a, b, c, N);
    benchmark::DoNotOptimize(c);
  }
  delete[](a);
  delete[](b);
  delete[](c);
}
BENCHMARK(BM_max_Normal)->RangeMultiplier(8)->Range(1<<8, 1<<24)->Complexity();

/* Max with SSE */
void arr_max_sse(int *__restrict__ a, int *__restrict__ b, int *__restrict__ c, int N) {
  for (int i = 0; i < N/4; i++) {
    __m128i av = _mm_loadu_si128((__m128i_u *)(a + i));
    __m128i bv = _mm_loadu_si128((__m128i_u *)(b + i));
    __m128i cv = _mm_max_epi32(av, bv);
    _mm_storeu_si128((__m128i_u *)(c + i), cv);
  }
}

static void BM_max_SSE(benchmark::State &state) {
  int N = state.range(0);
  int *a = new int[N];
  int *b = new int[N];
  int *c = new int[N];
  for (int i = 0; i < N; i++) {
    a[i] = rand() - RAND_HALF;
    b[i] = rand() - RAND_HALF;
  }

  for (auto _ : state) {
    arr_max_sse(a, b, c, N);
    benchmark::DoNotOptimize(c);
  }
  delete[](a);
  delete[](b);
  delete[](c);
}
BENCHMARK(BM_max_SSE)->RangeMultiplier(8)->Range(1<<8, 1<<24)->Complexity();

/* Max AVX */
void arr_max_avx(int *__restrict__ a, int *__restrict__ b, int *__restrict__ c, int N) {
  for (int i = 0; i < N/8; i++) {
    __m256i av = _mm256_loadu_si256((__m256i_u *)(a + i));
    __m256i bv = _mm256_loadu_si256((__m256i_u *)(b + i));
    __m256i cv = _mm256_max_epi32(av, bv);
    _mm256_storeu_si256((__m256i_u *)(c + i), cv);
  }
}

static void BM_max_AVX(benchmark::State &state) {
  int N = state.range(0);
  int *a = new int[N];
  int *b = new int[N];
  int *c = new int[N];
  for (int i = 0; i < N; i++) {
    a[i] = rand() - RAND_HALF;
    b[i] = rand() - RAND_HALF;
  }

  for (auto _ : state) {
    arr_max_avx(a, b, c, N);
    benchmark::DoNotOptimize(c);
  }
  delete[](a);
  delete[](b);
  delete[](c);
}
BENCHMARK(BM_max_AVX)->RangeMultiplier(8)->Range(1<<8, 1<<24)->Complexity();

