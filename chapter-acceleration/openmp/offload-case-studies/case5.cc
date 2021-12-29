#include <cmath>
#include <cstdio>

// static const int waveletsize = 3;
// static const int size = 6;

class HaarWavelet {
public:
  static const int waveletsize = 3;
  static const int size = 6;
  double wavelet[waveletsize] = {0.0, 1.0 / 2.0, 1.0 / 2.0};
  double sigmafactors[size + 1] = {1.00000000000, 7.07167810e-1, 5.00000000e-1,
                                   3.53553391e-1, 2.50000000e-1, 1.76776695e-1,
                                   1.25000000e-1};
  int* arr;
  HaarWavelet() {
    int *a = new int[10];
    arr = a;
    #pragma omp target enter data map(alloc:a)
  }

  ~HaarWavelet() {
    int *a = arr;
    #pragma omp target exit data map(delete:a)
    delete [] arr;
  }

#pragma omp declare target
  int getNumScales(int length) {
    return 1 + int(log(double(length - 1) / double(size - 1)) / M_LN2);
  }
#pragma omp end declare target

#pragma omp declare target
  int getMaxSize(int scale) { return int(pow(2, scale - 1)) * (size - 1) + 1; }
#pragma omp end declare target
};

#pragma omp declare target
void offload_driver(HaarWavelet *haar) {
  printf("Max Size: %d\n", haar->getMaxSize(1));
  printf("Num Scales: %d\n", haar->getNumScales(400));
}
#pragma omp end declare target

void test_offload_class() {
  HaarWavelet haar;

#pragma omp target map(to: haar.wavelet[:haar.waveletsize]) \
                    map(to:haar.sigmafactors[:haar.size + 1])
  {
    printf("Scales %d\n", haar.getNumScales(10));
    offload_driver(&haar);
  }
}