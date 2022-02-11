#include <assert.h>
#include <stdio.h>
#define N 2500

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error : %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void alloc_kernel(const float *in, float *out, int size) {
  int arr[N];
  unsigned int ystride = blockDim.y * gridDim.y;
  unsigned int xstride = blockDim.x * gridDim.x;
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;

  for (unsigned int y = tidy; y < size; y += ystride) {
    for (unsigned int x = tidx; x < size; x += xstride) {
      out[y * size + x] = in[y * size + x] * 1.0;
    }
  }
}

int main() {
  int size = 1000;
  int elems = size * size;
  float *in;
  float *out;

  checkCuda(cudaMallocManaged(&in, elems * sizeof(float)));
  checkCuda(cudaMallocManaged(&out, elems * sizeof(float)));

  for (int i = 0; i < elems; i++) {
    in[i] = i * 1.0;
  }

  cudaEvent_t start, stop;

  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  // printf("Starting\n");

  checkCuda(cudaEventRecord(start));

  dim3 blocksize(16, 16, 1);
  dim3 gridsize(16, 16, 1);

  alloc_kernel<<<gridsize, blocksize>>>(in, out, size);

  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));

  float sum = 0.0f;
  for (int i = 0; i < elems; i++) {
    sum += out[i];
  }
  printf("Sum %2.2f\n", sum);

  cudaFree(in);
  cudaFree(out);
}
