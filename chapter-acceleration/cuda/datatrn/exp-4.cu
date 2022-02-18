#include <assert.h>
#include <cstdlib>
#include <stdio.h>

#ifndef BLOCKSIZEX
#define BLOCKSIZEX 16
#endif

#ifndef GRIDSIZEX
#define GRIDSIZEX 8
#endif

#ifndef BLOCKSIZEY
#define BLOCKSIZEY 16
#endif

#ifndef GRIDSIZEY
#define GRIDSIZEY 8
#endif

// const int bxlen = 25;
// const int bylen = 25;

__global__ void alloc_test(const float *in, float *out, int xlimit, int ylimit,
                           int bxlen, int bylen) {
  unsigned int ystride = blockDim.y * gridDim.y;
  unsigned int xstride = blockDim.x * gridDim.x;

  float *tmp = (float *)malloc(sizeof(float) * bxlen * bylen);

  for (unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; y < ylimit;
       y += ystride) {
    for (unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; x < xlimit;
         x += xstride) {

      if (tmp == NULL) {
        printf("Allocation failed\n");
        return;
      }
    }
  }

  free(tmp);
}

int main(int argc, char **argv) {

  if (argc < 4) {
    fprintf(stderr, "Usage \n\t ./exp <mem> <gridx> <gridy> <blockx> <blocky> [bxlen] [bylen]");
  }

  int mem = atoi(argv[1]);

  int gridx = atoi(argv[2]);
  int gridy = atoi(argv[3]);

  int blockx = atoi(argv[4]);
  int blocky = atoi(argv[5]);

  int bxlen = 25;
  int bylen = 25;

  if (argc == 8) {
    bxlen = atoi(argv[6]);
    bylen = atoi(argv[7]);
  }

  // cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
  // printf("Limit %lu\n", limit);
  int xlimit = 1000;
  int ylimit = 1000;
  float *in, *out;

  cudaMallocManaged(&in, sizeof(float) * xlimit * ylimit);
  cudaMallocManaged(&out, sizeof(float) * xlimit * ylimit);

  for (int y = 0; y < ylimit; y++) {
    for (int x = 0; x < xlimit; x++) {
      int idx = (y * xlimit + x);
      in[idx] = idx * 1.0;
    }
  }

  dim3 blockDim(blockx, blocky, 1);
  dim3 gridDim(gridx, gridy, 1);
  int threads = blockDim.x * blockDim.y;

  size_t shared_mem = mem * 1024 * 1024;
  printf("Setting Malloc Heap size %lu\n", shared_mem);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, shared_mem);

  // 1M per block
  alloc_test<<<gridDim, blockDim>>>(in, out, xlimit, ylimit, bxlen, bylen);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error execting kernel %s\n", cudaGetErrorString(error));
    assert(error == cudaSuccess);
  }

  cudaDeviceSynchronize();

#ifdef DEBUG
  printf("\nOutput\n");
  for (int y = 0; y < ylimit; y++) {
    for (int x = 0; x < xlimit; x++) {
      int idx = (y * xlimit + x);
      printf("%f ", out[idx]);
    }
    printf("\n");
  }
  printf("\n");
#endif

  cudaFree(in);
  cudaFree(out);
}
