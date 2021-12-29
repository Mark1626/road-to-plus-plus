#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#pragma omp declare target
inline void swap(float &a, float &b) {
  float temp = a;
  a = b;
  b = temp;
}

float qselect(float *arr, int len, int nth) {
  int start = 0;
  for (int index = 0; index < len - 1; index++) {
    if (arr[index] > arr[len - 1])
      continue;
    swap(arr[index], arr[start]);
    start++;
  }
  swap(arr[len - 1], arr[start]);

  if (nth == start)
    return arr[start];

  return start > nth ? qselect(arr, start, nth)
                     : qselect(arr + start, len - start, nth - start);
}
#pragma omp end declare target

void test_offload_sort(int N) {
  std::srand(std::time(0));
  float a = 5.0;
  float *values = (float *)malloc(N * sizeof(float));

  printf("Offload example\n\n");
  // printf("Unsorted array\n");
  for (int i = 0; i < N; i++) {
    auto val = (float)std::rand() / (float)(RAND_MAX / a);
    values[i] = val;
    // printf("%f ", values[i]);
  }
  // printf("\n\n");

  #pragma omp target map(to : N) map(tofrom : values[:N])
  {
    // Perform sort in one thread of one team
    #pragma omp teams
    {
      int team_id = omp_get_team_num();
      if (team_id == 1) {

        #pragma omp parallel
        {
          #pragma omp master
          {
            #pragma omp critical
            {
              float mid = qselect(values, N, N/2);

              printf("Mid %f\n", mid);
              // printf("Partially sorted array within GPU\n");
              // for (int i = 0; i < N; i++) {
              //   printf("%f ", values[i]);
              // }
              // printf("\n");
            }
          }
          #pragma omp barrier
          printf("Median found, can perform other actions\n");
        }
      }
    }
  }

  // printf("Sorted array\n");
  // for (int i = 0; i < N; i++) {
  //   printf("%f ", values[i]);
  // }
  // printf("\n");

  free(values);
}

// Reference OpenMP implementation
void test_parallel_region_sort(int N) {
  std::srand(std::time(0));
  float a = 5.0;
  float *values = (float *)malloc(N * sizeof(float));

  printf("\n\nOpenMP example\n\n");
  printf("Unsorted array\n");
  for (int i = 0; i < N; i++) {
    auto val = (float)std::rand() / (float)(RAND_MAX / a);
    values[i] = val;
    printf("%f ", values[i]);
  }
  printf("\n");

#pragma omp parallel shared(values, N)
  {
    // Perform sort in one thread of one team
    #pragma omp master
    {
      #pragma omp critical
        {
          printf("Sorting now.....\n\n");
          float mid = qselect(values, N, N/2);
          printf("Mid %f\n", mid);
        }
    }
    #pragma omp barrier
    printf("Median found, can perform other actions\n");
}

printf("Sorted array\n");
for (int i = 0; i < N; i++) {
  printf("%f ", values[i]);
}
printf("\n");

  free(values);
}
