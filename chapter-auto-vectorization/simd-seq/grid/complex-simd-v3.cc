#pragma once
#include "comm.hh"

#include "simd-complex-v3.hh"

void gridding_simd_v3(Matrix<CFloat> &grid, Matrix<CFloat> &convFunc,
                   const CFloat &cVis, const int iu, const int iv,
                   const int support) {
  simdv3::complex2<float> cvis_vec = {
      .d = {cVis.real(), cVis.imag(), cVis.real(), cVis.imag()}};

  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    int uoff = 0;

    simdv3::complex2<float> *conv_vec =
        reinterpret_cast<simdv3::complex2<float> *>(&convFunc(voff, uoff));
    simdv3::complex2<float> *grid_vec = reinterpret_cast<simdv3::complex2<float> *>(
        &grid(iv + suppv, iu - support));

    // 2n+1, the last one can be done separately
    for (int i = 0; i < support; i++) {
      simdv3::grid(grid_vec[i], conv_vec[i], cvis_vec);
    }

    // Last grid point
    int suppu = support;
    uoff = suppu + support;
    CFloat wt = convFunc(voff, uoff);
    grid(iv + suppv, iu + suppu) += cVis * wt;
  }
}

inline void gridding_simd_v3_4(Matrix<CFloat> &grid, Matrix<CFloat> &convFunc,
                     const CFloat &cVis, const int iu, const int iv,
                     const int support) {
  simdv3::complex4<float> cvis_vec = {.d = {cVis.real(), cVis.imag(), cVis.real(),
                                          cVis.imag(), cVis.real(), cVis.imag(),
                                          cVis.real(), cVis.imag()}};

  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    int uoff = 0;

    simdv3::complex4<float> *conv_vec =
        reinterpret_cast<simdv3::complex4<float> *>(&convFunc(voff, uoff));
    simdv3::complex4<float> *grid_vec = reinterpret_cast<simdv3::complex4<float> *>(
        &grid(iv + suppv, iu - support));

    int rem = (2 * support + 1) % 4;
    int tiles = (2 * support + 1) / 4;

    // 2n+1, the last one can be done separately
    for (int i = 0; i < tiles; i++) {
      simdv3::grid(grid_vec[i], conv_vec[i], cvis_vec);
    }

    for (int i = 1; i <= rem; i++) {
      // Last grid point
      int suppu = support - rem + i;
      uoff = suppu + support;
      CFloat wt = convFunc(voff, uoff);
      grid(iv + suppv, iu + suppu) += cVis * wt;
    }
  }
}

#define BENCH

#ifdef BENCH
void BM_grid_simd_v3_4(benchmark::State &state) {
  int N = state.range(0);
  int convN = state.range(1);

  for (auto _ : state) {
    gridder(N, convN, gridding_simd_v3_4);
  }
}
#endif


#ifdef BENCH
void BM_grid_simd_v3(benchmark::State &state) {
  int N = state.range(0);
  int convN = state.range(1);

  for (auto _ : state) {
    gridder(N, convN, gridding_simd_v3);
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
