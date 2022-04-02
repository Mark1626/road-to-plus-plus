#pragma once
#include "comm.hh"

#include <immintrin.h>
#include <xmmintrin.h>

namespace simd {
template <typename T> struct complex4 { std::complex<T> d[4]; };

template <> struct complex4<float> { alignas(32) std::complex<float> d[4]; };

#define MASK(a, b, c, d) ((a << 6) + (b << 4) + (c << 2) + d)

// b may be simplified as it's a constant
// c += a * b
void grid(complex4<float> &c, complex4<float> &a, complex4<float> &b) {
  // ra1 ia1 ra2 ia2 ra3 ia3 ra4 ia4
  float *a_raw = reinterpret_cast<float(&)[8]>(a);
  // rb1 ib1 rb2 ib2 rb3 ib3 rb4 ib4
  float *b_raw = reinterpret_cast<float(&)[8]>(b);
  float *c_raw = reinterpret_cast<float(&)[8]>(c);

  __m256 a_vec = _mm256_loadu_ps(a_raw);
  __m256 b_vec = _mm256_loadu_ps(b_raw);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2 ra3rb3 ia3ib3 ra4ia4 ia4ib4
  __m256 interm1 = _mm256_mul_ps(a_vec, b_vec);

  const float sign[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
  const int sample1 = 0b01010101;
  const int sample2 = 0b10101010;
  const int swap_mask = MASK(2, 3, 0, 1);

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

  __m256 c_vec = _mm256_loadu_ps(c_raw);

  c_vec = _mm256_add_ps(c_vec, res_vec);

  _mm256_storeu_ps(c_raw, res_vec);
}

} // namespace simd

void gridding_simd_4(Matrix<CFloat> &grid, Matrix<CFloat> &conv,
                     const CFloat &cVis, const int iu, const int iv,
                     const int support) {
  simd::complex4<float> cvis_vec = {.d = {cVis, cVis, cVis, cVis}};

  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    int uoff = 0;

    simd::complex4<float> *conv_vec =
        reinterpret_cast<simd::complex4<float> *>(&conv(voff, uoff));
    simd::complex4<float> *grid_vec = reinterpret_cast<simd::complex4<float> *>(
        &grid(iv + suppv, iu - support));
    ;

    // 2n+1, the last one can be done separately
    for (int i = 0; i < support/2; i++) {
      simd::grid(grid_vec[i], conv_vec[i], cvis_vec);
    }

    // Last grid point
    int suppu = support;
    uoff = suppu + support;
    CFloat wt = conv(voff, uoff);
    grid(iv + suppv, iu + suppu) += cVis * wt;
  }
}

#ifdef BENCH
void BM_grid_simd_4(benchmark::State &state) {
  int N = state.range(0);
  int convN = state.range(1);

  for (auto _ : state) {
    gridder(N, convN, gridding_simd_4);
  }
}
#endif

#ifdef DEBUG
int main() {
  int N = 16;
  int convN = 8;
  gridder(N, convN, gridding_simd_4);
}
#endif
