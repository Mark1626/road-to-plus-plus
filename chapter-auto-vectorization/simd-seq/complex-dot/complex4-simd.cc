#include <benchmark/benchmark.h>
#include <complex>
#include <immintrin.h>
#include <xmmintrin.h>

namespace simd {
template <typename T> struct complex4 { std::complex<T> d[4]; };

template <> struct complex4<float> { _Alignas(32) std::complex<float> d[4]; };

#define MASK_8(a, b, c, d, e, f, g, h) ((a<<14) + (b << 12) + (c << 10) + (d << 8) + (e << 6) + (f << 4) + (g << 2) + h)
#define MASK_4(a, b, c, d) ((a << 6) + (b << 4) + (c << 2) + d)

void dotprod(complex4<float> &c, complex4<float> &a, complex4<float> &b) {
  // ra1 ia1 ra2 ia2 ra3 ia3 ra4 ia4
  float *a_raw = reinterpret_cast<float(&)[8]>(a);
  // rb1 ib1 rb2 ib2 rb3 ib3 rb4 ib4
  float *b_raw = reinterpret_cast<float(&)[8]>(b);
  float res_raw[8];

  __m256 a_vec = _mm256_loadu_ps(a_raw);
  __m256 b_vec = _mm256_loadu_ps(b_raw);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2 ra3rb3 ia3ib3 ra4ia4 ia4ib4
  __m256 interm1 = _mm256_mul_ps(a_vec, b_vec);

  const float sign[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
  const int sample1 = 0b01010101;
  const int sample2 = 0b10101010;
  // TODO: Check endianness
  const int swap_mask = MASK_4(2, 3, 0, 1);

  __m256 twist1 = _mm256_blend_ps(a_vec, b_vec, sample1);
  __m256 twist2 = _mm256_blend_ps(a_vec, b_vec, sample2);

  twist1 = _mm256_permute_ps(twist1, swap_mask);

  // ra1ia1 ra2ia2 rb1ib1 rb2ib2
  __m256 interm2 = _mm256_mul_ps(twist1, twist2);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2
  // ra1ia1 ra2ia2 rb1ib1 rb2ib2

  // ra1ia1 ia1ib1 rb1ib1 ia2ib2
  __m256 interm3 = _mm256_blend_ps(interm1, interm2, sample1);
  // ra1rb1 ra2ia2 ra2ia2 rb2ib2
  __m256 interm4 = _mm256_blend_ps(interm1, interm2, sample2);

  // ia1ib1 ra1ia1 ia2ib2 rb1ib1
  interm3 = _mm256_permute_ps(interm3, swap_mask);

  __m256 sign_vec = _mm256_load_ps(sign);
  // -ia1ib1 ra1ia1 -ia2ib2 rb1ib1
  interm3 = _mm256_mul_ps(interm3, sign_vec);

  __m256 res_vec = _mm256_add_ps(interm3, interm4);

  _mm256_storeu_ps(res_raw, res_vec);

  // This is not needed
  // c = reinterpret_cast<complex4<float>(&)>(res_raw);
}
} // namespace simd

inline void dotprod_simd4_stdcomplex_to_stdcomplex(
    std::complex<float> *cArr, std::complex<float> *aArr,
    std::complex<float> *bArr, const int N) {

  simd::complex4<float> *a_vec = reinterpret_cast<simd::complex4<float> *>(aArr);
  simd::complex4<float> *b_vec = reinterpret_cast<simd::complex4<float> *>(bArr);
  simd::complex4<float> *c_vec = reinterpret_cast<simd::complex4<float> *>(cArr);

  for (int i = 0; i < N/4; i++) {
    simd::dotprod(c_vec[i], a_vec[i], b_vec[i]);
  }
}

void BM_dotprod_simd4_stdcomplex_to_stdcomplex(benchmark::State &state) {
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
    dotprod_simd4_stdcomplex_to_stdcomplex(cArr, aArr, bArr, N);

    benchmark::DoNotOptimize(cArr);
  }

  delete[] aArr;
  delete[] bArr;
  delete[] cArr;
}

