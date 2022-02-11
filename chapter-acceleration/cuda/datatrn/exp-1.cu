/*
  nvcc -o exp-1 exp-1.cu

  Run with
  ./exp-1 trad
  Testing traditional memory transfer
  Elapsed time 56.385632
  
  ./exp-1 trad async
  Testing traditional memory transfer Async
  Elapsed time 55.350655
  
  ./exp-1 managed
  Testing managed memory
  Elapsed time 10.263168

  # Prefetch 100% data
  ./exp-1 managed prefetch 1
  Testing managed memory prefetch
  Elapsed time 1.202976

  # Prefetch 50% data
  ./exp-1 managed prefetch 2
  Testing managed memory prefetch
  Elapsed time 4.863584

*/

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int N = 1 << 25;
const int blockSize = 256;

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error : %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

template <typename T> static __global__ void grid_1D(T *ptr, size_t size) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t n = size / sizeof(T);
  //   printf("%lu %d %d %lu %lu\n", tid, gridDim.x, blockDim.x, stride, n);

  for (int i = tid; i < n; i += stride) {
    ptr[tid] = 2.0;
  }
}

static void test_Transfer() {
  printf("Testing traditional memory transfer\n");
  int numBlocks = (N + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));

  checkCuda(cudaEventRecord(start));

  float *in;
  float *d_in;

  in = new float[N];

  checkCuda(cudaMalloc(&d_in, N * sizeof(float)));
  checkCuda(cudaMemcpy(d_in, in, N * sizeof(float), cudaMemcpyHostToDevice));

  grid_1D<float><<<numBlocks, blockSize>>>(d_in, N * sizeof(float));

  checkCuda(cudaMemcpy(in, d_in, N * sizeof(float), cudaMemcpyDeviceToHost));

  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));

  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Elapsed time %f\n", elapsedTime);

  for (int i = 0; i < N; i++) {
    if (fabs(in[i] - 2.0) > 0.0001) {
      printf("Assertion failed for %d exp: %f act: %f \n", i, in[i], 2.0);
      break;
    }
  }

  checkCuda(cudaFree(d_in));
  delete[] in;
  checkCuda(cudaEventDestroy(start));
  checkCuda(cudaEventDestroy(stop));
}

static void test_Transfer_Async() {
  printf("Testing traditional memory transfer Async\n");
  int numBlocks = (N + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));

  checkCuda(cudaEventRecord(start));

  float *in;
  float *d_in;

  in = new float[N];

  checkCuda(cudaMalloc(&d_in, N * sizeof(float)));
  checkCuda(cudaMemcpyAsync(d_in, in, N * sizeof(float), cudaMemcpyHostToDevice));

  grid_1D<float><<<numBlocks, blockSize>>>(d_in, N * sizeof(float));

  checkCuda(cudaMemcpyAsync(in, d_in, N * sizeof(float), cudaMemcpyDeviceToHost));

  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));

  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Elapsed time %f\n", elapsedTime);

  for (int i = 0; i < N; i++) {
    if (fabs(in[i] - 2.0) > 0.0001) {
      printf("Assertion failed for %d exp: %f act: %f \n", i, in[i], 2.0);
      break;
    }
  }

  checkCuda(cudaFree(d_in));
  delete[] in;
  checkCuda(cudaEventDestroy(start));
  checkCuda(cudaEventDestroy(stop));
}

static void test_Managed_Prefetch(int factor) {
  printf("Testing managed memory prefetch\n");
  int numBlocks = (N + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));

  checkCuda(cudaEventRecord(start));

  int device = -1;
  checkCuda(cudaGetDevice(&device));

  float *in;
  checkCuda(cudaMallocManaged(&in, N * sizeof(float)));
  checkCuda(cudaMemPrefetchAsync(in, (N / factor) * sizeof(float), device, NULL));

  grid_1D<float><<<numBlocks, blockSize>>>(in, N * sizeof(float));

  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));

  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Elapsed time %f\n", elapsedTime);

  for (int i = 0; i < N; i++) {
    if (fabs(in[i] - 2.0) > 0.0001) {
      printf("Assertion failed for %d exp: %f act: %f \n", i, in[i], 2.0);
      break;
    }
  }

  checkCuda(cudaFree(in));

  checkCuda(cudaEventDestroy(start));
  checkCuda(cudaEventDestroy(stop));
}

static void test_Managed() {
  printf("Testing managed memory\n");
  int numBlocks = (N + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));

  checkCuda(cudaEventRecord(start));

  float *in;
  checkCuda(cudaMallocManaged(&in, N * sizeof(float)));

  grid_1D<float><<<numBlocks, blockSize>>>(in, N * sizeof(float));

  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));

  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Elapsed time %f\n", elapsedTime);

  for (int i = 0; i < N; i++) {
    if (fabs(in[i] - 2.0) > 0.0001) {
      printf("Assertion failed for %d exp: %f act: %f \n", i, in[i], 2.0);
      break;
    }
  }

  checkCuda(cudaFree(in));

  checkCuda(cudaEventDestroy(start));
  checkCuda(cudaEventDestroy(stop));
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Number of arguments must be 2\n");
    return 1;
  }

  if (strcmp(argv[1], "managed") == 0) {
    if (argc == 4 && strcmp(argv[2], "prefetch") == 0) {
      int factor = atoi(argv[3]);
      test_Managed_Prefetch(factor);
    } else {
      test_Managed();
    }
  } else if (strcmp(argv[1], "trad") == 0) {
    if (argc == 3 && strcmp(argv[2], "async") == 0) {
      test_Transfer_Async();
    } else {
      test_Transfer();
    }
  } else {
    fprintf(stderr, "Unknown option\n");
    return 1;
  }
}