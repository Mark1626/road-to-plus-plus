#pragma once
#include "comm.hh"

#include "simd-complex.hh"

void gridding_simd(Matrix<CFloat> &grid, Matrix<CFloat> &convFunc,
                   const CFloat &cVis, const int iu, const int iv,
                   const int support) {
  simd::complex2<float> cvis_vec = {
      .d = {cVis.real(), cVis.imag(), cVis.real(), cVis.imag()}};

  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    int uoff = 0;

    simd::complex2<float> *conv_vec =
        reinterpret_cast<simd::complex2<float> *>(&convFunc(voff, uoff));
    simd::complex2<float> *grid_vec = reinterpret_cast<simd::complex2<float> *>(
        &grid(iv + suppv, iu - support));

    // 2n+1, the last one can be done separately
    for (int i = 0; i < support; i++) {
      simd::grid(grid_vec[i], conv_vec[i], cvis_vec);
    }

    // Last grid point
    int suppu = support;
    uoff = suppu + support;
    CFloat wt = convFunc(voff, uoff);
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
