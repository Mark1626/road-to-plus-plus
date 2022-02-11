/*
  nvcc -o exp-3 exp-3.cu -O3 -Xcompiler -fopenmp -DASSERT
  ./exp2 2000 25

  This algorithm is not optimized for SIMT architecture, and runs into memory
  issues for larger window sizes

  Image size 100 100
  Window size 5 5
  Time taken CPU Grid: 4 ms
  threads: (16 16 1) blocks: (16 16 1)
  Elapsed GPU time 29.144129 ms
  threads: (16 16 1) blocks: (16 16 1)
  Elapsed GPU time Managed Memory 38.452671 ms
  Assertions complete
  Assertions complete managed
*/

#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <stdio.h>

using namespace std::chrono;

const int NDIM = 2;
const int TOLERANCE = 0.001;

#define bxlen 50
#define bylen 50

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

void sliding_window(const float *arr, const bool *mask, float *res1,
                    float *res2, int shape[NDIM], int resShape[NDIM]) {
  const int xlen = shape[0];

  printf("%d %d\n", resShape[0], resShape[1]);

#pragma omp parallel for collapse(2)
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      // printf("y: %d x: %d\n", y, x);
      int pos[NDIM];
      int blc[NDIM];
      int trc[NDIM];

      pos[0] = x;
      pos[1] = y;

      blc[0] = pos[0];
      blc[1] = pos[1];

      trc[0] = blc[0] + bxlen;
      trc[1] = blc[1] + bylen;

      float *tmp = new float[bxlen * bylen];
      int len = 0;

      for (int yy = blc[1]; yy < trc[1]; yy++) {
        for (int xx = blc[0]; xx < trc[0]; xx++) {
          int idx = yy * xlen + xx;
          if (mask[idx]) {
            tmp[len] = arr[idx];
            len++;
          }
        }
      }
      unsigned long nth = len / 2;

      float mid = qselect(tmp, len, nth);

      if (len % 2 == 0) {
        mid += qselect(tmp, len, nth - 1);
        mid /= 2.0;
      }

      // float mid = median(tmp, len);

      // printf("Here %f \n", mid);

      res1[y * resShape[0] + x] = mid;
      res2[y * resShape[0] + x] = mid;

      delete[] tmp;
    }
  }
}

__global__ void gliding_window(const float *arr, const bool *mask, float *res1,
                               float *res2, const unsigned int xlen,
                               const unsigned int ylen,
                               const unsigned int xlimit,
                               const unsigned int ylimit) {

  float tmp[bxlen * bylen];
  // float tmp[bxlen * bylen];
  unsigned int ystride = blockDim.y * gridDim.y;
  unsigned int xstride = blockDim.x * blockDim.x;
  // printf("xstride: %d ystride: %d\n", xstride, ystride);

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

      trc[0] = blc[0] + bxlen;
      trc[1] = blc[1] + bylen;

      // float mid = gliding_median(arr, blc, trc, xlen, ylen, bxlen, bylen);
      int len = 0;

      // float sum = 0.0;
      // #pragma unroll
      for (int yy = blc[1]; yy < trc[1]; yy++) {
        for (int xx = blc[0]; xx < trc[0]; xx++) {
          int idx = yy * xlen + xx;
          if (mask[idx]) {
            tmp[len] = arr[idx];
            len++;
          }
        }
      }
      unsigned long nth = len / 2;

      // float mid = median(tmp, len);
      float mid = qselect(tmp, len, nth);

      if (len % 2 == 0) {
        mid += qselect(tmp, len, nth - 1);
        mid /= 2.0;
      }

      res1[y * xlimit + x] = mid;
      res2[y * xlimit + x] = mid;
    }
  }
}

void gpu_sliding_window(const float *arr, const bool *mask, float *res1,
                        float *res2, int shape[NDIM], int resShape[NDIM],
                        int blockxsz, int blockysz, int gridxsize,
                        int gridysize) {

  size_t arrSize = sizeof(float) * shape[0] * shape[1];
  size_t maskSize = sizeof(bool) * shape[0] * shape[1];
  size_t resSize = sizeof(float) * resShape[0] * resShape[1];

#ifndef GDEBUG
  dim3 dimBlock(blockxsz, blockysz, 1);
#else
  dim3 dimBlock(1, 1, 1);
#endif

  int ylimit = resShape[1];
  int xlimit = resShape[0];

#ifndef GDEBUG
  int gridx = (xlimit + dimBlock.x - 1) / dimBlock.x;
  int gridy = (ylimit + dimBlock.y - 1) / dimBlock.y;
  // dim3 dimGrid(gridx, gridy, 1);
  dim3 dimGrid(gridxsize, gridysize, 1);
#else
  dim3 dimGrid(1, 1, 1);
#endif

  printf("threads: (%d %d %d) blocks: (%d %d %d)\n", dimBlock.x, dimBlock.y,
         dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);

  float *d_arr;
  bool *d_mask;
  float *d_res_1;
  float *d_res_2;

  cudaEvent_t start, stop;

  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  // printf("Starting\n");

  checkCuda(cudaEventRecord(start));
  // printf("Starting 2\n");

  checkCuda(cudaMalloc(&d_mask, maskSize));
  checkCuda(cudaMalloc(&d_res_1, resSize));
  checkCuda(cudaMalloc(&d_arr, arrSize));
  checkCuda(cudaMalloc(&d_res_2, resSize));

  checkCuda(cudaMemcpyAsync(d_mask, mask, maskSize, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpyAsync(d_arr, arr, arrSize, cudaMemcpyHostToDevice));

  // Run the kernel
  gliding_window<<<dimGrid, dimBlock>>>(d_arr, d_mask, d_res_1, d_res_2,
                                        shape[0], shape[1], resShape[0],
                                        resShape[1]);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error execting kernel %s\n", cudaGetErrorString(error));
    assert(error == cudaSuccess);
  }

  // printf("Finished kernel\n");

  // cudaDeviceSynchronize();

  // printf("Synched\n");

  checkCuda(cudaMemcpyAsync(res1, d_res_1, resSize, cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpyAsync(res2, d_res_2, resSize, cudaMemcpyDeviceToHost));

  // printf("Copy back\n");

  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));

  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Elapsed GPU time %f ms\n", elapsedTime);

  checkCuda(cudaFree(d_arr));
  checkCuda(cudaFree(d_res_1));
  checkCuda(cudaFree(d_mask));
  checkCuda(cudaFree(d_res_2));

  checkCuda(cudaEventDestroy(start));
  checkCuda(cudaEventDestroy(stop));
}

void experiment(int SIZE, int blockxsz, int blockysz, int gridxsz,
                int gridysz) {
  int shape[NDIM] = {SIZE, SIZE};
  int resShape[NDIM] = {shape[0] - bxlen, shape[1] - bylen};

  size_t arrSize = sizeof(float) * shape[0] * shape[1];
  size_t maskSize = sizeof(bool) * shape[0] * shape[1];
  size_t resSize = sizeof(float) * resShape[0] * resShape[1];

  size_t ndim = 2;

  float *arr = (float *)malloc(arrSize);
  bool *mask = (bool *)malloc(maskSize);

  double a = 5.0;

  for (int y = 0; y < shape[1]; y++) {
    for (int x = 0; x < shape[0]; x++) {
      float val = (double)std::rand() / (double)(RAND_MAX / a);
      arr[y * shape[0] + x] = val;
      mask[y * shape[0] + x] = val > 1.0;
    }
  }

#ifdef DEBUG
  printf("Data\n");
  printf("------------------------------\n");
  printf("\nArray\n");
  for (int y = 0; y < shape[1]; y++) {
    for (int x = 0; x < shape[0]; x++) {
      float val = arr[y * resShape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }

  printf("\nMask\n");
  for (int y = 0; y < shape[1]; y++) {
    for (int x = 0; x < shape[0]; x++) {
      bool val = mask[y * resShape[0] + x];
      printf("%d ", val);
    }
    printf("\n");
  }

#endif

  float *res_cpu_1 = (float *)malloc(sizeof(float) * resShape[0] * resShape[1]);
  float *res_gpu_m_1 =
      (float *)malloc(sizeof(float) * resShape[0] * resShape[1]);
  float *res_gpu_1 = (float *)malloc(sizeof(float) * resShape[0] * resShape[1]);

  float *res_cpu_2 = (float *)malloc(sizeof(float) * resShape[0] * resShape[1]);
  float *res_gpu_m_2 =
      (float *)malloc(sizeof(float) * resShape[0] * resShape[1]);
  float *res_gpu_2 = (float *)malloc(sizeof(float) * resShape[0] * resShape[1]);

  printf("Image size %d %d\n", SIZE, SIZE);
  printf("Window size %d %d\n", bxlen, bylen);

  auto t1 = high_resolution_clock::now();

  sliding_window(arr, mask, res_cpu_1, res_cpu_2, shape, resShape);

  auto t2 = high_resolution_clock::now();
  printf("Time taken CPU Grid: %ld ms\n",
         duration_cast<milliseconds>(t2 - t1).count());

  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 32 * 1024 * 1024);

  gpu_sliding_window(arr, mask, res_gpu_1, res_gpu_2, shape, resShape, blockxsz,
                     blockysz, gridxsz, gridysz);

  // gpu_sliding_window_managed(arr, mask, res_gpu_m_1, res_gpu_m_2, shape,
  // hboxsz,
  //                            resShape);

#ifdef DEBUG
  printf("Result 1\n");
  printf("------------------------------\n");
  printf("\nExpected\n");
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float val = res_cpu_1[y * resShape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }

  printf("\nActual\n");
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float val = res_gpu_1[y * resShape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }

  printf("\nActual Managed\n");
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float val = res_gpu_m_1[y * resShape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }
  printf("\nResult 2\n");
  printf("------------------------------\n");
  printf("\nExpected\n");
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float val = res_cpu_2[y * resShape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }

  printf("\nActual\n");
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float val = res_gpu_2[y * resShape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }

  printf("\nActual Managed\n");
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float val = res_gpu_m_2[y * resShape[0] + x];
      printf("%f ", val);
    }
    printf("\n");
  }
#endif

#ifdef ASSERT
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float actual = res_gpu_1[y * resShape[0] + x];
      float expected = res_cpu_1[y * resShape[0] + x];
      if (fabs(actual - expected) > TOLERANCE) {
        fprintf(stderr,
                "Assertion failed value 1 at %d %d expected: %f actual: %f\n",
                x, y, expected, actual);
      }

      actual = res_gpu_2[y * resShape[0] + x];
      expected = res_cpu_2[y * resShape[0] + x];
      if (fabs(actual - expected) > TOLERANCE) {
        fprintf(stderr,
                "Assertion failed value 2 at %d %d expected: %f actual: %f\n",
                x, y, expected, actual);
      }
    }
  }
  printf("Assertions complete\n");
  /*
  for (int y = 0; y < resShape[1]; y++) {
    for (int x = 0; x < resShape[0]; x++) {
      float actual = res_gpu_m_1[y * resShape[0] + x];
      float expected = res_cpu_1[y * resShape[0] + x];
      if (fabs(actual - expected) > TOLERANCE) {
        fprintf(stderr,
                "Assertion failed value 1 at %d %d expected: %f actual: %f\n",
                x, y, expected, actual);
      }

      actual = res_gpu_m_2[y * resShape[0] + x];
      expected = res_cpu_2[y * resShape[0] + x];
      if (fabs(actual - expected) > TOLERANCE) {
        fprintf(stderr,
                "Assertion failed value 2 at %d %d expected: %f actual: %f\n",
                x, y, expected, actual);
      }
    }
  }
  printf("Assertions complete managed\n");
  */
#endif

  free(res_gpu_m_1);
  free(res_gpu_1);
  free(res_cpu_1);
  free(res_gpu_m_2);
  free(res_gpu_2);
  free(res_cpu_2);
  free(arr);
  free(mask);
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr,
            "Usage:\n exp-8 <SIZE> <blockx> <blocky> <gridx> <gridy>\n");
    return 1;
  }
  int SIZE = std::atoi(argv[1]);
  int blockxsz = std::atoi(argv[2]);
  int blockysz = std::atoi(argv[3]);
  int gridxsz = std::atoi(argv[4]);
  int gridysz = std::atoi(argv[5]);

  experiment(SIZE, blockxsz, blockysz, gridxsz, gridysz);
}
