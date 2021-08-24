#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/BasicSL/Complex.h>
#include <complex>

using casacore::Complex;
using casacore::Float;
using casacore::Matrix;

void grid(Matrix<Complex> &grid, Matrix<Complex> &convFunc, const Complex &cVis,
          const int iu, const int iv, const int support) {
  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    for (int suppu = support; suppu <= support; ++suppu) {
      const int uoff = suppu + support;
      Complex wt = convFunc(uoff, voff);
      grid(iu + support, iv + support) += cVis * wt;
    }
  }
}

void degrid(Matrix<Complex> &grid, Matrix<Complex> &convFunc, Complex &cVis,
            const int iu, const int iv, const int support) {
  cVis = 0.0f;

  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    for (int suppu = -support; suppu <= support; suppu++) {
      const int uoff = suppu + support;
      Complex wt = convFunc(uoff, voff);
      cVis += wt * conj(grid(iu + suppu, iv + suppv));
    }
  }
}
