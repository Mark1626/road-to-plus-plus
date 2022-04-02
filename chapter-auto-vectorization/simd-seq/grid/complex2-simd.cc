#pragma once
#include "comm.hh"

#include <immintrin.h>
#include <xmmintrin.h>

namespace simd {
template <typename T> struct complex2 { std::complex<T> d[2]; };

template <> struct complex2<float> { alignas(16) std::complex<float> d[2]; };

#define MASK(a, b, c, d) ((a << 6) + (b << 4) + (c << 2) + d)

// b may be simplified as it's a constant
// c += a * b
void grid(complex2<float> &c, complex2<float> &a, complex2<float> &b) {
  // ra1 ia1 ra2 ia2
  float *a_raw = reinterpret_cast<float(&)[4]>(a);
  // rb1 ib1 rb2 ib2
  float *b_raw = reinterpret_cast<float(&)[4]>(b);
  float *c_raw = reinterpret_cast<float(&)[4]>(c);

  __m128 a_vec = _mm_loadu_ps(a_raw);
  __m128 b_vec = _mm_loadu_ps(b_raw);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2
  __m128 interm1 = _mm_mul_ps(a_vec, b_vec);

  const float sign[4] = {-1.0, 1.0, -1.0, 1.0};
  const int sample1 = 0b0101;
  const int sample2 = 0b1010;
  const int swap_mask = MASK(2, 3, 0, 1);

  __m128 twist1 = _mm_blend_ps(a_vec, b_vec, sample1);
  __m128 twist2 = _mm_blend_ps(a_vec, b_vec, sample2);

  twist1 = _mm_permute_ps(twist1, swap_mask);

  // ra1ia1 ra2ia2 rb1ib1 rb2ib2
  __m128 interm2 = _mm_mul_ps(twist1, twist2);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2
  // ra1ia1 ra2ia2 rb1ib1 rb2ib2

  // ra1ia1 ia1ib1 rb1ib1 ia2ib2
  __m128 interm3 = _mm_blend_ps(interm1, interm2, sample1);
  // ra1rb1 ra2ia2 ra2ia2 rb2ib2
  __m128 interm4 = _mm_blend_ps(interm1, interm2, sample2);

  // ia1ib1 ra1ia1 ia2ib2 rb1ib1
  interm3 = _mm_permute_ps(interm3, swap_mask);

  __m128 sign_vec = _mm_loadu_ps(sign);
  // -ia1ib1 ra1ia1 -ia2ib2 rb1ib1
  interm3 = _mm_mul_ps(interm3, sign_vec);

  __m128 res_vec = _mm_add_ps(interm3, interm4);

  __m128 c_vec = _mm_loadu_ps(c_raw);

  c_vec = _mm_add_ps(c_vec, res_vec);

  _mm_storeu_ps(c_raw, c_vec);
}
} // namespace simd

void gridding_simd(Matrix<CFloat> &grid, Matrix<CFloat> &conv,
                   const CFloat &cVis, const int iu, const int iv,
                   const int support) {
  simd::complex2<float> cvis_vec = {.d = {cVis, cVis}};

  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    int uoff = 0;

    simd::complex2<float> *conv_vec =
        reinterpret_cast<simd::complex2<float> *>(&conv(voff, uoff));
    simd::complex2<float> *grid_vec = reinterpret_cast<simd::complex2<float> *>(
        &grid(iv + suppv, iu - support));
    ;

    // 2n+1, the last one can be done separately
    for (int i = 0; i < support; i++) {
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
void BM_grid_simd(benchmark::State &state) {
  int N = state.range(0);
  int convN = state.range(1);

  for (auto _ : state) {
    gridder(N, convN, gridding_simd);
  }
}
#endif

#ifdef DEBUG
int main() {
  int N = 16;
  int convN = 8;
  gridder(N, convN, gridding_simd);
}
#endif
