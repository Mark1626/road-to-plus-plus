#include "comm.hh"
#include <assert.h>
#include <cstdio>

void gridding_ptr(Matrix<CFloat> &grid, Matrix<CFloat> &conv,
                  const CFloat &cVis, const int iu, const int iv,
                  const int support) {
  float rVis = cVis.real();
  float iVis = cVis.imag();
  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    const int uoff = 0;
    float *wtPtrF = reinterpret_cast<float *>(&conv(uoff, voff));
    float *gridPtrF =
        reinterpret_cast<float *>(&grid(iu - support, iv + suppv));
    for (int suppu = -support; suppu <= support;
         suppu++, wtPtrF += 2, gridPtrF += 2) {
      gridPtrF[0] += rVis * wtPtrF[0] - iVis * wtPtrF[1];
      gridPtrF[1] += rVis * wtPtrF[1] + iVis * wtPtrF[0];
    }
  }
}

void BM_grid_ptr(benchmark::State &state) {
  int N = state.range(0);
  int convN = state.range(1);

  for (auto _ : state) {
    gridder(N, convN, gridding_ptr);
  }
}
