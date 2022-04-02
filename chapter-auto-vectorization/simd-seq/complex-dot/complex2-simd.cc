#include <benchmark/benchmark.h>
#include <complex>
#include <immintrin.h>
#include <xmmintrin.h>

namespace simd {
template <typename T> struct complex2 { std::complex<T> d[2]; };

template <> struct complex2<float> { alignas(16) std::complex<float> d[2]; };

#define MASK(a, b, c, d) ((a << 6) + (b << 4) + (c << 2) + d)

void prod(complex2<float> &c, complex2<float> &a, complex2<float> &b) {
  // ra1 ia1 ra2 ia2
  float *a_raw = reinterpret_cast<float(&)[4]>(a);
  // rb1 ib1 rb2 ib2
  float *b_raw = reinterpret_cast<float(&)[4]>(b);
  float *c_raw = reinterpret_cast<float(&)[4]>(c);

  __m128 a_vec = _mm_load_ps(a_raw);
  __m128 b_vec = _mm_load_ps(b_raw);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2
  __m128 interm1 = _mm_mul_ps(a_vec, b_vec);

  const float sign[4] = {-1.0, 1.0, -1.0, 1.0};
  const int sample1 = 0b0101;
  const int sample2 = 0b1010;
  const int swap_mask = MASK(2, 3, 0, 1);

  __m128 twist1 = _mm_permute_ps(b_vec, swap_mask);
  // ra1ia2 ra2ia1 rb1ib2 rb2ib1
  __m128 interm2 = _mm_mul_ps(a_vec, twist1);

  // ra1ia1 ia1ib1 rb1ib1 ia2ib2
  __m128 interm3 = _mm_blend_ps(interm1, interm2, sample1);
  // ra1rb1 ra2ia2 ra2ia2 rb2ib2
  __m128 interm4 = _mm_blend_ps(interm1, interm2, sample2);

  // ia1ib1 ra1ia1 ia2ib2 rb1ib1
  interm3 = _mm_permute_ps(interm3, swap_mask);

  __m128 sign_vec = _mm_load_ps(sign);
  // -ia1ib1 ra1ia1 -ia2ib2 rb1ib1
  interm3 = _mm_mul_ps(interm3, sign_vec);

  __m128 res_vec = _mm_add_ps(interm3, interm4);

  _mm_store_ps(c_raw, res_vec);
}
} // namespace simd

inline void dotprod_simd_stdcomplex_to_stdcomplex(
    std::complex<float> *cArr, std::complex<float> *aArr,
    std::complex<float> *bArr, const int N) {

  simd::complex2<float> *a_vec = reinterpret_cast<simd::complex2<float> *>(aArr);
  simd::complex2<float> *b_vec = reinterpret_cast<simd::complex2<float> *>(bArr);
  simd::complex2<float> *c_vec = reinterpret_cast<simd::complex2<float> *>(cArr);

  for (int i = 0; i < N/2; i++) {
    simd::prod(c_vec[i], a_vec[i], b_vec[i]);
  }
}

void BM_dotprod_simd_stdcomplex_to_stdcomplex(benchmark::State &state) {
  // std::array<std::complex<float>, N> a;
  int N = state.range(0);

  std::complex<float> *aArr;
  std::complex<float> *bArr;
  std::complex<float> *cArr;

  aArr = new std::complex<float>[N];
  bArr = new std::complex<float>[N];
  cArr = new std::complex<float>[N];

  float a = 5.0;

  for (int i = 0; i < N; i++) {
    auto real = (double)std::rand() / (double)(RAND_MAX / a);
    auto imag = (double)std::rand() / (double)(RAND_MAX / a);
    aArr[i] = std::complex<float>(real, imag);

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr[i] = std::complex<float>(real, imag);
  }

  for (auto _ : state) {
    dotprod_simd_stdcomplex_to_stdcomplex(cArr, aArr, bArr, N);

    benchmark::DoNotOptimize(cArr);
  }

  delete[] aArr;
  delete[] bArr;
  delete[] cArr;
}

