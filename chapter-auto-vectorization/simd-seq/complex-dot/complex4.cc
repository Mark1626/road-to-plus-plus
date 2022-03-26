#pragma once
#include <complex>
#include <benchmark/benchmark.h>

template <typename T> struct complex4 {
  T imag[4];
  T real[4];
};

inline void dotprod_float4_to_stdcomplex(std::complex<float> *cArr,
                                         const complex4<float> *aArr,
                                         const complex4<float> *bArr,
                                         const int N) {
  #pragma omp simd
  for (int i = 0; i < N / 4; i++) {
    int idx = 4 * i;
    cArr[idx] = std::complex<float>((aArr[i].real[0] * bArr[i].real[0]) -
                                        (aArr[i].imag[0] * bArr[i].imag[0]),
                                    (aArr[i].real[0] * bArr[i].imag[0]) +
                                        (aArr[i].imag[0] * bArr[i].real[0]));

    cArr[idx + 1] =
        std::complex<float>((aArr[i].real[1] * bArr[i].real[1]) -
                                (aArr[i].imag[1] * bArr[i].imag[1]),
                            (aArr[i].real[1] * bArr[i].imag[1]) +
                                (aArr[i].imag[1] * bArr[i].real[1]));
    
    cArr[idx + 2] = std::complex<float>((aArr[i].real[2] * bArr[i].real[2]) -
                                        (aArr[i].imag[2] * bArr[i].imag[2]),
                                    (aArr[i].real[2] * bArr[i].imag[2]) +
                                        (aArr[i].imag[2] * bArr[i].real[2]));

    cArr[idx + 3] =
        std::complex<float>((aArr[i].real[3] * bArr[i].real[3]) -
                                (aArr[i].imag[3] * bArr[i].imag[3]),
                            (aArr[i].real[3] * bArr[i].imag[3]) +
                                (aArr[i].imag[3] * bArr[i].real[3]));
  }
}

void BM_dotprod_complex4_to_stdcomplex(benchmark::State &state) {
  int N = state.range(0);

  int N_quad = N / 4;

  complex4<float> *aArr;
  complex4<float> *bArr;
  std::complex<float> *cArr;

  aArr = new complex4<float>[N_quad];
  bArr = new complex4<float>[N_quad];
  cArr = new std::complex<float>[N];

  float a = 5.0;

  for (int i = 0; i < N_quad; i++) {
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
    aArr[i].real[2] = real;
    aArr[i].imag[2] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    aArr[i].real[3] = real;
    aArr[i].imag[3] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr[i].real[0] = real;
    bArr[i].imag[0] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr[i].real[1] = real;
    bArr[i].imag[1] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr[i].real[2] = real;
    bArr[i].imag[2] = imag;

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr[i].real[3] = real;
    bArr[i].imag[3] = imag;
  }

  for (auto _ : state) {
    dotprod_float4_to_stdcomplex(cArr, aArr, bArr, N);
    benchmark::DoNotOptimize(cArr);
  }

  delete[] aArr;
  delete[] bArr;
  delete[] cArr;
}
