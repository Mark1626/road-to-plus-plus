#include <stdio.h>
#include <assert.h>

#define BUFFERX 64
#define BUFFERY 64
#define WINDOWX 16
#define WINDOWY 16

#define NDIM 2

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error : %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}


__global__ void mem(const float *in, float *out, int xlimit, int ylimit,
                    int xlen, int ylen) {
  __shared__ float buffer[BUFFERX][BUFFERY];

  unsigned int ystride = blockDim.y * gridDim.y;
  unsigned int xstride = blockDim.x * gridDim.x;

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;

  for (unsigned int y = tidy; y < ylimit; y += ystride) {
    for (unsigned int x = tidx; x < xlimit; x += xstride) {
      
      if (tidy == 0 && tidx == 0) {
        for (int yy = 0; yy < BUFFERX; yy++) {
          #pragma unroll
          for (int xx = 0; xx < BUFFERY; xx++) {
            buffer[yy][xx] = in[((y + yy) * xlen) + (x + xx)];
          }
        }
      }
      __syncthreads();

      int pos[NDIM];
      int blc[NDIM];
      int trc[NDIM];

      pos[0] = tidx;
      pos[1] = tidy;

      blc[0] = pos[0];
      blc[1] = pos[1];

      trc[0] = blc[0] + WINDOWX;
      trc[1] = blc[1] + WINDOWY;

      float tmp[WINDOWX * WINDOWY];

      int len = 0;
      for (int yy=0; yy < WINDOWY; yy++) {
        #pragma unroll
        for (int xx=0; xx < WINDOWX; xx++) {
          tmp[len++] = buffer[yy][xx];
        }
      }

      // Reverse the order to simulate the memory bandwidth
      int m = WINDOWY * WINDOWX;
      for (int idx=0; idx < x; idx++) {
          float t = tmp[idx];
          tmp[idx] = tmp[m - idx];
          tmp[m - idx] = t;
      }

      float sum = 0.0;
      for (int idx=0; idx < WINDOWY; idx++) {
          sum += tmp[idx];
      }

      out[y * xlimit + x] = sum;

      __syncthreads();
    }
  }
}

int main() {
  int xlen = 1024;
  int ylen = 1024;
  int xlimit = xlen - WINDOWX;
  int ylimit = ylen - WINDOWY;

  float *in;
  float *out;

  cudaEvent_t start, stop;

  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));

  checkCuda(cudaEventRecord(start));

  checkCuda(cudaMallocManaged(&in, xlen * ylen * sizeof(float)));
  checkCuda(cudaMallocManaged(&out, xlimit * ylimit * sizeof(float)));

  for (int y = 0; y < ylen; y++) {
    for (int x = 0; x < xlen; x++) {
      in[y * xlen + x] = (y * xlen + x) * 1.0f;
      // out[y * xlen + x]
    }
  }

  for (int y = 0; y < ylimit; y++) {
    for (int x = 0; x < xlimit; x++) {
      out[y * xlimit + x] = 0.0f;
    }
  }

  dim3 numThreads(16, 16, 1);
  dim3 numBlocks(16, 16, 1);

  mem<<<numBlocks, numThreads>>>(in, out, xlimit, ylimit, xlen, ylen);

  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));

  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Elapsed GPU time %f ms\n", elapsedTime);

  // for (int y = 0; y < ylimit; y++) {
  //   for (int x = 0; x < xlimit; x++) {
  //     printf("%f ", out[y * xlimit + x]);
  //   }
  //   printf("\n");
  // }

  checkCuda(cudaFree(in));
  checkCuda(cudaFree(out));
}
