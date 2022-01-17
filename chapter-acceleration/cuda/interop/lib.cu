// #include "lib.cuh"
#include <cstddef>
#include <cstdio>

__global__ void saxpy_kernel(int N, float *a, float *b, float *c) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if (index < N) {
    c[index] = a[index] * 5.0 + b[index];
  }
}

__global__ void hello() {
  printf("Hello World\n");
}

void saxpy(int N, float *a, float* b, float *c) {
  float* d_a;
  float* d_b;
  float* d_c;
  size_t arrSize = N * sizeof(float);

  cudaMalloc(&d_a, arrSize);
  cudaMalloc(&d_b, arrSize);
  cudaMalloc(&d_c, arrSize);

  cudaMemcpy(d_a, a, arrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, arrSize, cudaMemcpyHostToDevice);

  int threads = 128;
  int blocksPerGrid = (N + threads - 1) / threads;

  saxpy_kernel <<< blocksPerGrid, threads >>> (N, d_a, d_b, d_c);

  cudaDeviceSynchronize();

  cudaMemcpy(c, d_c, arrSize, cudaMemcpyDeviceToHost);

  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);
}
