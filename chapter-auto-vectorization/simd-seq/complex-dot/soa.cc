#pragma once
#include <complex>
#include <benchmark/benchmark.h>

template <typename T> struct SOAComplex {
  T *real;
  T *imag;
};

inline void dotprod_complexfloat_to_float(SOAComplex<float> &cArr,
                                          const SOAComplex<float> &aArr,
                                          const SOAComplex<float> &bArr,
                                          const int N) {
  for (int i = 0; i < N; i++) {
    cArr.real[i] =
        (aArr.real[i] * bArr.real[i]) - (aArr.imag[i] * bArr.imag[i]);
    cArr.imag[i] =
        (aArr.real[i] * bArr.imag[i]) + (aArr.imag[i] * bArr.real[i]);
  }
}

void BM_dotprod_complexfloat_to_float(benchmark::State &state) {
  int N = state.range(0);

  SOAComplex<float> aArr;
  SOAComplex<float> bArr;
  SOAComplex<float> cArr;

  aArr.real = new float[N];
  aArr.imag = new float[N];

  bArr.real = new float[N];
  bArr.imag = new float[N];

  cArr.real = new float[N];
  cArr.imag = new float[N];

  float a = 5.0;

  for (int i = 0; i < N; i++) {
    auto real = (double)std::rand() / (double)(RAND_MAX / a);
    auto imag = (double)std::rand() / (double)(RAND_MAX / a);
    aArr.real[i] = real;
    aArr.imag[i] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr.real[i] = real;
    bArr.imag[i] = imag;
  }

  for (auto _ : state) {
    dotprod_complexfloat_to_float(cArr, aArr, bArr, N);

    benchmark::DoNotOptimize(cArr);
  }

  delete[] aArr.real;
  delete[] aArr.imag;
  delete[] bArr.real;
  delete[] bArr.imag;
  delete[] cArr.real;
  delete[] cArr.imag;
}

inline void dotprod_complexfloat_to_stdcomplex(std::complex<float> *cArr,
                                               const SOAComplex<float> &aArr,
                                               const SOAComplex<float> &bArr,
                                               const int N) {
  for (int i = 0; i < N; i++) {
    cArr[i] = std::complex<float>(
        (aArr.real[i] * bArr.real[i]) - (aArr.imag[i] * bArr.imag[i]),
        (aArr.real[i] * bArr.imag[i]) + (aArr.imag[i] * bArr.real[i]));
  }
}

void BM_dotprod_complexfloat_to_stdcomplex(benchmark::State &state) {
  int N = state.range(0);

  SOAComplex<float> aArr;
  SOAComplex<float> bArr;
  std::complex<float> *cArr;

  aArr.real = new float[N];
  aArr.imag = new float[N];

  bArr.real = new float[N];
  bArr.imag = new float[N];

  cArr = new std::complex<float>[N];

  float a = 5.0;

  for (int i = 0; i < N; i++) {
    auto real = (double)std::rand() / (double)(RAND_MAX / a);
    auto imag = (double)std::rand() / (double)(RAND_MAX / a);
    aArr.real[i] = real;
    aArr.imag[i] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr.real[i] = real;
    bArr.imag[i] = imag;
  }

  for (auto _ : state) {
    dotprod_complexfloat_to_stdcomplex(cArr, aArr, bArr, N);

    benchmark::DoNotOptimize(cArr);
  }

  delete[] aArr.real;
  delete[] aArr.imag;
  delete[] bArr.real;
  delete[] bArr.imag;
  delete[] cArr;
}

inline void dotprod_complexfloat_to_stdcomplex_simd(std::complex<float> *cArr,
                                               const SOAComplex<float> &aArr,
                                               const SOAComplex<float> &bArr,
                                               const int N) {
  #pragma omp simd
  for (int i = 0; i < N; i++) {
    cArr[i] = std::complex<float>(
        (aArr.real[i] * bArr.real[i]) - (aArr.imag[i] * bArr.imag[i]),
        (aArr.real[i] * bArr.imag[i]) + (aArr.imag[i] * bArr.real[i]));
  }
}

void BM_dotprod_complexfloat_to_stdcomplex_simd(benchmark::State &state) {
  int N = state.range(0);

  SOAComplex<float> aArr;
  SOAComplex<float> bArr;
  std::complex<float> *cArr;

  aArr.real = new float[N];
  aArr.imag = new float[N];

  bArr.real = new float[N];
  bArr.imag = new float[N];

  cArr = new std::complex<float>[N];

  float a = 5.0;

  for (int i = 0; i < N; i++) {
    auto real = (double)std::rand() / (double)(RAND_MAX / a);
    auto imag = (double)std::rand() / (double)(RAND_MAX / a);
    aArr.real[i] = real;
    aArr.imag[i] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr.real[i] = real;
    bArr.imag[i] = imag;
  }

  for (auto _ : state) {
    dotprod_complexfloat_to_stdcomplex_simd(cArr, aArr, bArr, N);

    benchmark::DoNotOptimize(cArr);
  }

  delete[] aArr.real;
  delete[] aArr.imag;
  delete[] bArr.real;
  delete[] bArr.imag;
  delete[] cArr;
}
