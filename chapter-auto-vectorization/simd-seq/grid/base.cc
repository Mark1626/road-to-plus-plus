#include "comm.hh"
#include <assert.h>
#include <cstdio>

void gridding(Matrix<CFloat> &grid, Matrix<CFloat> &conv, const CFloat &cVis,
              const int iu, const int iv, const int support) {
  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    for (int suppu = -support; suppu <= support; suppu++) {
      const int uoff = suppu + support;

      // printf("Conv idx %d %d iu %d iv %d\n", uoff, voff, iu, iv);
      // printf("Grid point %d %d\n", iu + suppu, iv + suppv);

      CFloat wt = conv(uoff, voff);
      grid(iu + suppu, iv + suppv) += cVis * wt;
    }
  }
}

void BM_grid_std_complex(benchmark::State &state) {
  int N = state.range(0);
  int convN = state.range(1);

  for (auto _ : state) {
    gridder(N, convN, gridding);
  }
}

// int main() {
//   int N = 16;
//   int convN = 8;
//   gridder(N, convN, gridding);
// }
