/*
  Does not work for odd window sizes
*/
#include <assert.h>
#include <chrono>
#include <stdio.h>

using namespace std::chrono;

#define NDIM 2

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

#define BLOCKX 16
#define BLOCKY 16
#define GRIDX 16
#define GRIDY 16

#define BXLEN 2
#define BYLEN 2

#define XBUFFER (BLOCKX + BXLEN - 1)
#define YBUFFER (BLOCKY + BYLEN - 1)

// #define DEBUG
// #define ASSERT

__global__ void
shared_mem_kernel(const float *in, float *out, const unsigned int xlen,
                  const unsigned int ylen, const unsigned int xlimit,
                  const unsigned int ylimit) {

  // float tmp[BXLEN][BYLEN];
  __shared__ float buffer[XBUFFER][YBUFFER];

  // printf("Buffer dim: %d, %d\n", bufferx, buffery);

  unsigned int ystride = blockDim.y * gridDim.y;
  unsigned int xstride = blockDim.x * gridDim.x;

  #ifdef DEBUG
  printf("xstride: %d ystride: %d\n", xstride, ystride);
  #endif
  // printf("x: %d y: %d\n", blockIdx.x * blockDim.x + threadIdx.x,
  //        blockIdx.y * blockDim.y + threadIdx.y);
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;

#ifdef DEBUG
      printf("Array\n");
      for (int yy = 0; yy < ylen; yy++) {
        for (int xx = 0; xx < xlen; xx++) {
          printf("%f ", in[yy * xlen + xx]);
        }
        printf("\n");
      }
#endif

  for (unsigned int y = tidy; y < ylimit; y += ystride) {
    for (unsigned int x = tidx; x < xlimit; x += xstride) {

      if (tidy == 0 && tidx == 0) {
        // Load from global into shared memory
        for (int yy = 0; yy < YBUFFER; yy++) {
          for (int xx = 0; xx < XBUFFER; xx++) {
            // printf("(%d, %d) %d\n", xx, yy, ((y+yy) * xlen) + (x + xx));
            buffer[yy][xx] = in[((y+yy) * xlen) + (x + xx)];
          }
        }
      }
      __syncthreads();

#ifdef DEBUG
/*
      printf("Buffer value\n");
      for (int yy = 0; yy < YBUFFER; yy++) {
        for (int xx = 0; xx < XBUFFER; xx++) {
          printf("%f ", buffer[yy][xx]);
        }
        printf("\n");
      }
*/
#endif

      // printf("x: %d y: %d\n", x, y);
      int pos[NDIM];
      int blc[NDIM];
      int trc[NDIM];

      pos[0] = tidx;
      pos[1] = tidy;

      blc[0] = pos[0];
      blc[1] = pos[1];

      trc[0] = blc[0] + BXLEN;
      trc[1] = blc[1] + BYLEN;

#ifdef DEBUG
/*
      printf("Shared Mem (%d %d)\n", tidx, tidy);
      for (int yy = blc[1]; yy < trc[1]; yy++) {
        for (int xx = blc[0]; xx < trc[0]; xx++) {
          printf("%f ", buffer[yy][xx]);
        }
        printf("\n");
      }

      printf("Global (%d %d)\n", tidx, tidy);
      for (int yy = y; yy < y + BYLEN; yy++) {
        for (int xx = x; xx < x + BXLEN; xx++) {
          printf("%f ", in[yy * xlen + xx]);
        }
        printf("\n");
      }
*/
      printf("Mem Box: (%d, %d) (%d, %d)\n", blc[0], blc[1], trc[0], trc[1]);
      printf("Actual Box: (%d, %d) (%d, %d)\n", x, y, x + BXLEN, y + BYLEN);

      for (int ya = y, ys = blc[1]; ya < y + BYLEN; ya++, ys++) {
        for (int xa = x, xs = blc[0]; xa < x + BXLEN; xa++, xs++) {
          float shm = buffer[ys][xs];
          float global = in[ya * xlen + xa];
          if (fabs(global - shm) > 0.001) {
            printf("Mismatch of values expected: %f actual: %f (%d, %d)\n",
                   global, shm, tidx, tidy);
          } else {
            printf("No problem (%d, %d)\n", tidx, tidy);
          }
        }
      }

#endif

      float sum = 0.0;
      // #pragma unroll 4
      for (int yy = blc[1]; yy < trc[1]; yy++) {
        for (int xx = blc[0]; xx < trc[0]; xx++) {
          int idx = yy * XBUFFER + xx;
          // Read from buffer rather than global mem
          sum += buffer[yy][xx];
        }
      }

      out[y * xlimit + x] = sum;
      __syncthreads();
    }
  }
}

__global__ void
without_mem_kernel(const float *in, float *out, const unsigned int xlen,
                   const unsigned int ylen, const unsigned int xlimit,
                   const unsigned int ylimit) {
  unsigned int ystride = blockDim.y * gridDim.y;
  unsigned int xstride = blockDim.x * gridDim.x;

  // printf("xstride: %d ystride: %d\n", xstride, ystride);
  // printf("x: %d y: %d\n", blockIdx.x * blockDim.x + threadIdx.x,
  //        blockIdx.y * blockDim.y + threadIdx.y);

  for (unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; y < ylimit;
       y += ystride) {
    for (unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; x < xlimit;
         x += xstride) {
      // printf("x: %d y: %d\n", x, y);
      int pos[NDIM];
      int blc[NDIM];
      int trc[NDIM];

      pos[0] = x;
      pos[1] = y;

      blc[0] = pos[0];
      blc[1] = pos[1];

      trc[0] = blc[0] + BXLEN;
      trc[1] = blc[1] + BYLEN;

      // printf("Box: (%d, %d) (%d, %d)\n", blc[0], blc[1], trc[0], trc[1]);

      float sum = 0.0;
      // #pragma unroll
      for (int yy = blc[1]; yy < trc[1]; yy++) {
        for (int xx = blc[0]; xx < trc[0]; xx++) {
          int idx = yy * xlen + xx;
          sum += in[idx];
        }
      }

      out[y * xlimit + x] = sum;
    }
  }
}

void host_reference(const float *in, float *out, const unsigned int xlen,
                    const unsigned int ylen, const unsigned int xlimit,
                    const unsigned int ylimit) {

#pragma omp parallel for collapse(2)
  for (int y = 0; y < ylimit; y++) {
    for (int x = 0; x < xlimit; x++) {
      int pos[NDIM];
      int blc[NDIM];
      int trc[NDIM];

      pos[0] = x;
      pos[1] = y;

      blc[0] = pos[0];
      blc[1] = pos[1];

      trc[0] = blc[0] + BXLEN;
      trc[1] = blc[1] + BYLEN;

      float sum = 0.0;

      for (int yy = blc[1]; yy < trc[1]; yy++) {
        for (int xx = blc[0]; xx < trc[0]; xx++) {
          int idx = yy * xlen + xx;
          sum += in[idx];
        }
      }

      out[y * xlimit + x] = sum;
    }
  }
}

void gpu_run_kernel(
    const float *arr, float *res, int shape[NDIM], int resShape[NDIM],
    int hboxsz[NDIM],
    void kernel(const float *in, float *out, const unsigned int xlen,
                const unsigned int ylen, const unsigned int xlimit,
                const unsigned int ylimit)) {
  size_t arrSize = sizeof(float) * shape[0] * shape[1];
  size_t resSize = sizeof(float) * resShape[0] * resShape[1];

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
#ifndef GDEBUG
  dim3 blocks(BLOCKX, BLOCKY, 1);
  dim3 grids(GRIDX, GRIDY, 1);
#else
  dim3 blocks(2, 2, 1);
  dim3 grids(1, 1, 1);
#endif

  printf("blocksize: (%d, %d, %d) grid: (%d, %d, %d)\n", blocks.x, blocks.y,
         blocks.z, grids.x, grids.y, grids.z);

  // int shared_mem =
  //     (grids.x + hboxsz[0] - 1) * (grids.y + hboxsz[1] - 1) * sizeof(float) *
  //     2;
  int shared_mem = 16 * 1024;
  printf("Shared memory usage: %lu\n", shared_mem);
  // Run the kernel
  kernel<<<grids, blocks, shared_mem>>>(d_arr, d_res, shape[0], shape[1],
                                        resShape[0], resShape[1]);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error execting kernel %s\n", cudaGetErrorString(error));
    assert(error == cudaSuccess);
  }

  // printf("Finished kernel\n");

  // cudaDeviceSynchronize();

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

void experiment(int shape[NDIM], int hboxsz[NDIM], int mem) {
  int resShape[NDIM] = {shape[0] - hboxsz[0], shape[1] - hboxsz[1]};

  size_t arrSize = sizeof(float) * shape[0] * shape[1];
  size_t resSize = sizeof(float) * resShape[0] * resShape[1];

  float *arr = (float *)malloc(arrSize);

  double a = 5.0;

  for (int y = 0; y < shape[1]; y++) {
    for (int x = 0; x < shape[0]; x++) {
      float val = (double)std::rand() / (double)(RAND_MAX / a);
      arr[y * shape[0] + x] = y * shape[0] + x;
      // arr[y * shape[0] + x] = val;
    }
  }

#ifdef DEBUG
/*
  printf("Data\n");
  printf("------------------------------\n");
  printf("\nArray\n");
  for (int y = 0; y < shape[1]; y++) {
    for (int x = 0; x < shape[0]; x++) {
      float val = arr[y * shape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }
*/
#endif

  // size_t shared_mem = mem * 1024 * 1024;
  // printf("Setting Malloc Heap size %lu\n", shared_mem);
  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, shared_mem);

  float *res_cpu = (float *)malloc(sizeof(float) * resShape[0] * resShape[1]);
  float *res_gpu_without_smem =
      (float *)malloc(sizeof(float) * resShape[0] * resShape[1]);
  float *res_gpu = (float *)malloc(sizeof(float) * resShape[0] * resShape[1]);

  printf("Array size %d %d\n", shape[0], shape[1]);
  printf("Window size %d %d\n", hboxsz[0], hboxsz[1]);

  auto t1 = std::chrono::high_resolution_clock::now();

  host_reference(arr, res_cpu, shape[0], shape[1], resShape[0], resShape[1]);

  auto t2 = std::chrono::high_resolution_clock::now();
  printf("Time taken CPU Grid: %ld ms\n",
         duration_cast<std::chrono::milliseconds>(t2 - t1).count());

  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 32 * 1024 * 1024);

  printf("Running shared memory test\n");
  gpu_run_kernel(arr, res_gpu, shape, resShape, hboxsz, shared_mem_kernel);

  printf("Running without shared memory test\n");
  gpu_run_kernel(arr, res_gpu_without_smem, shape, resShape, hboxsz,
                 without_mem_kernel);

#ifdef DEBUG
  printf("Result 1\n");
  printf("------------------------------\n");
  printf("\nExpected\n");
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float val = res_cpu[y * resShape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }

  printf("\nActual\n");
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float val = res_gpu[y * resShape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }

  printf("\nActual without shared mem\n");
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float val = res_gpu_without_smem[y * resShape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }
#endif

#ifdef ASSERT
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float actual = res_gpu[y * resShape[0] + x];
      float expected = res_cpu[y * resShape[0] + x];
      if (fabs(actual - expected) > TOLERANCE) {
        fprintf(stderr,
                "Assertion failed value at %d %d expected: %f actual: %f\n", x,
                y, expected, actual);
      }
    }
  }
  printf("Assertions complete kernel\n");
  /*
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float actual = res_gpu_without_smem[y * resShape[0] + x];
      float expected = res_cpu[y * resShape[0] + x];
      if (fabs(actual - expected) > TOLERANCE) {
        fprintf(stderr,
                "Assertion failed value at %d %d expected: %f actual: %f\n",
                x, y, expected, actual);
      }
    }
  }
  */
  printf("Assertions complete for kernel with shared mem\n");
#endif

  free(res_gpu_without_smem);
  free(res_gpu);
  free(res_cpu);

  free(arr);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage:\n exp-9 <SIZE>\n");
    return 1;
  }
  int dim = std::atoi(argv[1]);
  int shape[NDIM] = {dim, dim};
  int hboxsz[NDIM] = {BXLEN, BYLEN};
  // int mem = std::atoi(argv[2]);
  int mem = 0; // this is not used

  experiment(shape, hboxsz, mem);
}
