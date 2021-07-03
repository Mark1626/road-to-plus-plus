#include <array>
#include <emmintrin.h>
#include <iostream>

#include <immintrin.h>

void case_study_1(int *__restrict__ a, int *__restrict__ b, int *__restrict__ d,
                  int N) {
  const int s = 10;
  #pragma omp simd
  for (int i = 0; i < N; i++) {
    // d contains distinct elements
    int j = d[i];
    a[j] += s * b[i];
  }
}

// #pragma omp ordered simd seems to be not supported by clang
// void case_study_2(int *__restrict__ a, int *__restrict__ b, int *__restrict__ d,
//                   int N) {
//   const int s = 10;
//   #pragma omp simd
//   for (int i = 0; i < N; i++) {
//     // d does not contains distinct elements
//     int j = d[i];
//     #pragma omp ordered simd
//     a[j] += s * b[i];
//   }
// }

int main() {}
