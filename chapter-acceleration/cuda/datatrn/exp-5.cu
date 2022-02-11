/*
  This code has occupancy 97%
  Yet the performance less than a CPU OpenMP version
*/
#include <assert.h>
#include <chrono>
#include <stdio.h>

using namespace std::chrono;

const int TOLERANCE = 0.001;

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error : %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

#define SWAP(a, b)                                                             \
  {                                                                            \
    float t = (a);                                                             \
    (a) = (b);                                                                 \
    (b) = t;                                                                   \
  }

__device__ __host__ float qselect(float *arr, int len, int nth) {
  int start = 0;
  for (int index = 0; index < len - 1; index++) {
    if (arr[index] > arr[len - 1])
      continue;
    SWAP(arr[index], arr[start]);
    start++;
  }
  SWAP(arr[len - 1], arr[start]);

  if (nth == start)
    return arr[start];

  return start > nth ? qselect(arr, start, nth)
                     : qselect(arr + start, len - start, nth - start);
}

__global__ void qselect_kernel(const float *in, float *out, int N, int limit,
                               int blen) {
  unsigned int xstride = blockDim.x * blockDim.x;

  float *tmp = (float *)malloc(sizeof(float) * blen);

  for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < limit;
       idx += xstride) {

    int st = idx;
    int en = idx + blen;

    if (tmp == NULL) {
      printf("Allocation failed\n");
      return;
    }

    int len = 0;
    for (int ii = st; ii < en; ii++) {

      tmp[len] = in[ii];
      len++;
    }
    unsigned long nth = len / 2;

    float mid = qselect(tmp, len, nth);

    if (len % 2 == 0) {
      mid += qselect(tmp, len, nth - 1);
      mid /= 2.0;
    }

    out[idx] = mid;
  }

  free(tmp);
}

__global__ void select_kernel(const float *in, float *out, int N, int limit,
                              int blen) {
  unsigned int xstride = blockDim.x * blockDim.x;

  float *tmp = (float *)malloc(sizeof(float) * blen);

  for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < limit;
       idx += xstride) {

    int st = idx;
    int en = idx + blen;

    if (tmp == NULL) {
      printf("Allocation failed\n");
      return;
    }

    int len = 0;
    for (int ii = st; ii < en; ii++) {
      tmp[len] = in[ii];
      len++;
    }
    unsigned long nth = len / 2;

    // Sort tmp with a basic select sort
    // This is assuming warps don't get diverged
    int min;
    for (int ii = 0; ii <= nth; ii++) {
      min = ii;
      for (int jj = ii; jj < len; jj++) {
        if (tmp[jj] < tmp[min]) {
          min = jj;
        }
      }

      float t = tmp[ii];
      tmp[ii] = tmp[min];
      tmp[min] = t;
    }

    // printf("DEBUG Kernel\n");
    // for (int ii = 0; ii < len; ii++) {
    //   printf("%f ", tmp[ii]);
    // }
    // printf("\n");

    float mid = tmp[nth];

    if (len % 2 == 0) {
      mid += tmp[nth - 1];
      mid /= 2.0;
    }

    out[idx] = mid;
  }

  free(tmp);
}

void host_reference(const float *in, float *out, int N, int limit, int blen) {

#pragma omp parallel for
  for (unsigned int idx = 0; idx < limit; idx += 1) {

    int st = idx;
    int en = idx + blen;

    float *tmp = (float *)malloc(sizeof(float) * blen);

    if (tmp == NULL) {
      printf("Allocation failed\n");
      // return;
    }

    int len = 0;
    for (int ii = st; ii < en; ii++) {

      tmp[len] = in[ii];
      len++;
    }

    unsigned long nth = len / 2;

    float mid = qselect(tmp, len, nth);

    if (len % 2 == 0) {
      mid += qselect(tmp, len, nth - 1);
      mid /= 2.0;
    }

    out[idx] = mid;
    free(tmp);
  }
}

#ifndef BLOCKSIZE
#define BLOCKSIZE 256
#endif

#ifndef GRIDSIZE
#define GRIDSIZE 1
#endif

void gpu_run_kernel(const float *arr, float *res, int N, int limit, int blen,
                    void kernel(const float *in, float *out, int N, int limit,
                                int blen)) {
  size_t arrSize = sizeof(float) * N;
  size_t resSize = sizeof(float) * limit;

  float *d_arr;
  float *d_res;

  cudaEvent_t start, stop;

  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  // printf("Starting\n");

  checkCuda(cudaEventRecord(start));
  // printf("Starting 2\n");

  checkCuda(cudaMalloc(&d_res, resSize));
  checkCuda(cudaMalloc(&d_arr, arrSize));

  checkCuda(cudaMemcpyAsync(d_arr, arr, arrSize, cudaMemcpyHostToDevice));

  // int grids = (N + BLOCKSIZE - 1) / BLOCKSIZE;
  int grids = 512;

  printf("blocksize: %d grid: %d\n", BLOCKSIZE, grids);

  // Run the kernel
  kernel<<<grids, BLOCKSIZE>>>(d_arr, d_res, N, limit, blen);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error execting kernel %s\n", cudaGetErrorString(error));
    assert(error == cudaSuccess);
  }

  // printf("Finished kernel\n");

  cudaDeviceSynchronize();

  // printf("Synched\n");

  checkCuda(cudaMemcpyAsync(res, d_res, resSize, cudaMemcpyDeviceToHost));

  // printf("Copy back\n");

  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));

  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Elapsed GPU time %f ms\n", elapsedTime);

  checkCuda(cudaFree(d_arr));
  checkCuda(cudaFree(d_res));

  checkCuda(cudaEventDestroy(start));
  checkCuda(cudaEventDestroy(stop));
}

#define ASSERT

void experiment(int N, int blen, int mem) {
  int limit = N - blen;

  size_t arrSize = sizeof(float) * N;
  size_t resSize = sizeof(float) * limit;

  float *arr = (float *)malloc(arrSize);

  double a = 5.0;

  for (int idx = 0; idx < N; idx++) {
    float val = (double)std::rand() / (double)(RAND_MAX / a);
    arr[idx] = val;
  }

#ifdef DEBUG
  printf("Data\n");
  printf("------------------------------\n");
  printf("\nArray\n");
  for (int idx = 0; idx < N; idx++) {
    float val = arr[idx];
    printf("%f ", val);
  }
  printf("\n");
#endif

  size_t shared_mem = mem * 1024 * 1024;
  printf("Setting Malloc Heap size %lu\n", shared_mem);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, shared_mem);


  float *res_cpu = (float *)malloc(sizeof(float) * limit);
  float *res_gpu_select = (float *)malloc(sizeof(float) * limit);
  float *res_gpu_qselect = (float *)malloc(sizeof(float) * limit);

  printf("Array size %d\n", N);
  printf("Window size %d\n", blen);

  auto t1 = std::chrono::high_resolution_clock::now();

  host_reference(arr, res_cpu, N, limit, blen);

  auto t2 = std::chrono::high_resolution_clock::now();
  printf("Time taken CPU Grid: %ld ms\n",
         duration_cast<std::chrono::milliseconds>(t2 - t1).count());

  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 32 * 1024 * 1024);

  printf("Running qselect test\n");
  gpu_run_kernel(arr, res_gpu_qselect, N, limit, blen, qselect_kernel);

  printf("Running select sort test\n");
  gpu_run_kernel(arr, res_gpu_select, N, limit, blen, select_kernel);

#ifdef DEBUG
  printf("Result 1\n");
  printf("------------------------------\n");
  printf("\nExpected\n");
  for (int idx = 0; idx < limit; idx++) {

    float val = res_cpu[idx];
    printf("%f ", val);
  }
  printf("\n");

  printf("\nActual\n");
  for (int idx = 0; idx < limit; idx++) {
    float val = res_gpu_qselect[idx];
    printf("%f ", val);
  }
  printf("\n");

  printf("\nActual select\n");
  for (int idx = 0; idx < limit; idx++) {
    float val = res_gpu_select[idx];
    printf("%f ", val);
  }
  printf("\n");
#endif

#ifdef ASSERT
  for (int idx = 0; idx < limit; idx++) {
    float actual = res_gpu_qselect[idx];
    float expected = res_cpu[idx];
    if (fabs(actual - expected) > TOLERANCE) {
      fprintf(stderr,
              "Assertion failed value 1 at %d expected: %f actual: %f\n", idx,
              expected, actual);
    }
  }
  printf("Assertions complete qselect\n");
  for (int idx = 0; idx < limit; idx++) {
    float actual = res_gpu_select[idx];
    float expected = res_cpu[idx];
    if (fabs(actual - expected) > TOLERANCE) {
      fprintf(stderr,
              "Assertion failed value 1 at %d expected: %f actual: %f\n",
              idx, expected, actual);
    }
  }
  printf("Assertions complete select\n");
#endif

  free(res_gpu_select);
  free(res_gpu_qselect);
  free(res_cpu);

  free(arr);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage:\n exp-2 <SIZE> <WINDOW_SIZE> <mem>\n");
    return 1;
  }
  int n = std::atoi(argv[1]);
  int blen = std::atoi(argv[2]);
  int mem = std::atoi(argv[3]);
  experiment(n, blen, mem);
}
