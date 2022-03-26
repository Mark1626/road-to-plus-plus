#include <benchmark/benchmark.h>
#include <complex>

inline void dotprod_stdcomplex_to_stdcomplex(std::complex<float> *cArr,
                                             const std::complex<float> *aArr,
                                             const std::complex<float> *bArr,
                                             const int N) {
  for (int i = 0; i < N; i++) {
    cArr[i] = aArr[i] * bArr[i];
  }
}

void BM_dotprod_stdcomplex_to_stdcomplex(benchmark::State &state) {
  // std::array<std::complex<float>, N> a;
  int N = state.range(0);

  std::complex<float> *aArr;
  std::complex<float> *bArr;
  std::complex<float> *cArr;

  aArr = new std::complex<float>[N];
  bArr = new std::complex<float>[N];
  cArr = new std::complex<float>[N];

  float a = 5.0;

  for (int i = 0; i < N; i++) {
    auto real = (double)std::rand() / (double)(RAND_MAX / a);
    auto imag = (double)std::rand() / (double)(RAND_MAX / a);
    aArr[i] = std::complex<float>(real, imag);

    real = (double)std::rand() / (double)(RAND_MAX / a);
    imag = (double)std::rand() / (double)(RAND_MAX / a);
    bArr[i] = std::complex<float>(real, imag);
  }

  for (auto _ : state) {
    dotprod_stdcomplex_to_stdcomplex(cArr, aArr, bArr, N);

    benchmark::DoNotOptimize(cArr);
  }

  delete[] aArr;
  delete[] bArr;
  delete[] cArr;
}
