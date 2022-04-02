#pragma once
#include <complex>
#ifdef BENCH
#include <benchmark/benchmark.h>
#endif

typedef std::complex<float> CFloat;

struct CFloat2 {
  alignas(16) std::complex<float> v[2];
};

template <typename T> class Matrix {
public:
  int M, N;
  T *val;
  Matrix<T>(int M, int N) : M(M), N(N) { val = new T[M * N]; }
  ~Matrix<T>() {
    if (val)
      delete[] val;
  }
  T &operator()(int x, int y) { return val[x * N + y]; }
};

void gridder(int N, int convN,
             void (*gridFn)(Matrix<CFloat> &, Matrix<CFloat> &, const CFloat &,
                            const int, const int, const int)) {
  int support = (convN / 2) - 1;

  float a = 5.0;
  Matrix<CFloat> conv(convN, convN);
  for (int i = 0; i < convN; i++) {
    for (int j = 0; j < convN; j++) {
      auto real = (double)std::rand() / (double)(RAND_MAX / a);
      auto imag = (double)std::rand() / (double)(RAND_MAX / a);
      conv(i, j) = CFloat(real, imag);
    }
  }

  Matrix<CFloat> grid(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      grid(i, j) = CFloat(0.0, 0.0);
    }
  }

  Matrix<CFloat> visibility(N, N);
  for (int i = 3; i < N - 3; i++) {
    for (int j = 3; j < N - 3; j++) {
      auto real = (double)std::rand() / (double)(RAND_MAX / a);
      auto imag = (double)std::rand() / (double)(RAND_MAX / a);
      visibility(i, j) = CFloat(real, imag);
    }
  }

  int offset = support;
  for (int u = 0; u < N - 2 * offset; u++) {
    for (int v = 0; v < N - 2 * offset; v++) {
      int iu = u + offset;
      int iv = v + offset;
      CFloat cvis = visibility(iu, iv);
      gridFn(grid, conv, cvis, u + offset, v + offset, support);
    }
  }

  #ifdef BENCH
  benchmark::DoNotOptimize(gridFn);
  #endif

#ifdef DEBUG
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      CFloat gridPt = grid(i, j);
      printf("(%.1f, %.1f) ", gridPt.real(), gridPt.imag());
    }
    printf("\n");
  }
#endif
}

