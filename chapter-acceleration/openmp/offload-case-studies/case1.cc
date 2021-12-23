#include <cstdio>
#include <cstdlib>

void test_sum_int(size_t N) {
  float *arr = new float[N];
  for (int i = 0; i < N; i++) {
    arr[i] = i;
  }

  int sum = 0;
  for (int i = 0; i < N; i++) {
    sum = sum + arr[i];
  }
  printf("Sum is %d sum\n", sum);

  sum = 0;
#pragma omp parallel for default(none) shared(arr, N) reduction(+ : sum)
  for (int i = 0; i < N; i++) {
#pragma omp critical
    { sum = sum + arr[i]; }
  }
  printf("Parallel Sum is %d sum\n", sum);

  sum = 0.0;
#pragma omp target map(to : arr[:N], N) map(tofrom : sum)
  {
#pragma omp teams distribute parallel for reduction(+ : sum)
    for (int i = 0; i < N; i++) {
#pragma omp critical
      { sum = sum + i; }
    }
  }
  printf("Offload Sum is %d sum\n", sum);

  delete[] arr;
}

void test_sum_float(size_t N) {
  float *arr = new float[N];
  float a = 5.0;

  srand(10);
  for (int i = 0; i < N; i++) {
    auto val = (float)std::rand() / (float)(RAND_MAX / a);
    arr[i] = val;
  }

  float sum = 0.0;
  for (int i = 0; i < N; i++) {
    sum = sum + arr[i];
  }
  printf("Sum is %f sum\n", sum);

  sum = 0.0;
#pragma omp parallel for default(none) shared(arr, N) reduction(+ : sum)
  for (int i = 0; i < N; i++) {
#pragma omp critical
    { sum = sum + arr[i]; }
  }
  printf("Parallel Sum is %f sum\n", sum);

  sum = 0.0;
#pragma omp target map(to : arr[:N], N) map(tofrom : sum)
  {
#pragma omp teams distribute parallel for reduction(+ : sum)
    for (int i = 0; i < N; i++) {
#pragma omp critical
      { sum = sum + arr[i]; }
    }
  }
  printf("Offload Sum is %f sum\n", sum);
  delete[] arr;
}
