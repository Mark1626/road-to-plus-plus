void case_study_1(int *__restrict__ a, int *__restrict__ b) {
  #pragma omp simd
  for (int i = 0; i < 64; i++) {
    a[i] *= b[i];
  }
}
