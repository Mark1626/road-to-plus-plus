#pragma once
#include <benchmark/benchmark.h>
#include <complex>

template <typename T> struct complex2 {
  alignas(8) T imag[2];
  alignas(8) T real[2];
};

inline void dotprod_complex2_to_stdcomplex(std::complex<float> *cArr,
                                           const complex2<float> *aArr,
                                           const complex2<float> *bArr,
                                           const int N) {
#pragma omp simd
  for (int i = 0; i < N / 2; i++) {
    int idx = 2 * i;
    cArr[idx] = std::complex<float>((aArr[i].real[0] * bArr[i].real[0]) -
                                        (aArr[i].imag[0] * bArr[i].imag[0]),
                                    (aArr[i].real[0] * bArr[i].imag[0]) +
                                        (aArr[i].imag[0] * bArr[i].real[0]));

    cArr[idx + 1] =
        std::complex<float>((aArr[i].real[1] * bArr[i].real[1]) -
                                (aArr[i].imag[1] * bArr[i].imag[1]),
                            (aArr[i].real[1] * bArr[i].imag[1]) +
                                (aArr[i].imag[1] * bArr[i].real[1]));
  }
}

void BM_dotprod_complex2_to_stdcomplex(benchmark::State &state) {
  int N = state.range(0);

  int N_half = N / 2;

  complex2<float> *aArr;
  complex2<float> *bArr;
  std::complex<float> *cArr;

  aArr = new complex2<float>[N_half];
  bArr = new complex2<float>[N_half];
  cArr = new std::complex<float>[N];

  float a = 5.0;

  for (int i = 0; i < N_half; i++) {
    auto real = (double)std::rand() / (double)(RAND_MAX / a);
    auto imag = (double)std::rand() / (double)(RAND_MAX / a);
    aArr[i].real[0] = real;
    aArr[i].imag[0] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    aArr[i].real[1] = real;
    aArr[i].imag[1] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr[i].real[0] = real;
    bArr[i].imag[0] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr[i].real[1] = real;
    bArr[i].imag[1] = imag;
  }

  for (auto _ : state) {
    dotprod_complex2_to_stdcomplex(cArr, aArr, bArr, N);
    benchmark::DoNotOptimize(cArr);
  }

  delete[] aArr;
  delete[] bArr;
  delete[] cArr;
}

inline void dotprod_complex2_to_complex2(complex2<float> *cArr,
                                         const complex2<float> *aArr,
                                         const complex2<float> *bArr,
                                         const int N) {

#pragma omp simd
  for (int i = 0; i < N; i++) {
    cArr[i].real[0] = (aArr[i].real[0] * bArr[i].real[0]) -
                      (aArr[i].imag[0] * bArr[i].imag[0]);
    cArr[i].real[1] = (aArr[i].real[1] * bArr[i].real[1]) -
                      (aArr[i].imag[1] * bArr[i].imag[1]);

    cArr[i].imag[0] = (aArr[i].real[0] * bArr[i].imag[0]) +
                      (aArr[i].imag[0] * bArr[i].real[0]);

    cArr[i].imag[1] = (aArr[i].real[1] * bArr[i].imag[1]) +
                      (aArr[i].imag[1] * bArr[i].real[1]);
  }
}

void BM_dotprod_complex2_to_complex2(benchmark::State &state) {
  int N = state.range(0);

  int N_half = N / 2;

  complex2<float> *aArr;
  complex2<float> *bArr;
  complex2<float> *cArr;

  aArr = new complex2<float>[N_half];
  bArr = new complex2<float>[N_half];
  cArr = new complex2<float>[N_half];

  float a = 5.0;

  for (int i = 0; i < N_half; i++) {
    auto real = (double)std::rand() / (double)(RAND_MAX / a);
    auto imag = (double)std::rand() / (double)(RAND_MAX / a);
    aArr[i].real[0] = real;
    aArr[i].imag[0] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    aArr[i].real[1] = real;
    aArr[i].imag[1] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr[i].real[0] = real;
    bArr[i].imag[0] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr[i].real[1] = real;
    bArr[i].imag[1] = imag;
  }

  for (auto _ : state) {
    dotprod_complex2_to_complex2(cArr, aArr, bArr, N_half);
    benchmark::DoNotOptimize(cArr);
  }

  delete[] aArr;
  delete[] bArr;
  delete[] cArr;
}
