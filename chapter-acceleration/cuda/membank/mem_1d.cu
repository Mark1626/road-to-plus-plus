#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define ALIGN_IDX(blk, bxlen, bank, idx)                                       \
  ((32 * bxlen * blk) + (32 * idx) + bank)

__device__ void align_sort(float *buffer, int limit, int bxlen) {

  for (int blk = 0; blk < (limit / 32) + 1; blk++) {

    for (int tid = 0; tid < 32; tid++) {
      int nth = bxlen / 2;
      int min;

      // printf("Mid: %d\n", nth);
      // Generic sort algorithm, could be replace with any other algorithm
      for (int ii = 0; ii <= nth; ii++) {
        min = ii;
        // printf("%d %d\n", ALIGN_IDX(tid, 0), ALIGN_IDX(tid, 1));
        for (int jj = ii; jj < bxlen; jj++) {
          if (buffer[ALIGN_IDX(blk, bxlen, tid, jj)] <
              buffer[ALIGN_IDX(blk, bxlen, tid, min)]) {
            min = jj;
          }
        }

        float t = buffer[ALIGN_IDX(blk, bxlen, tid, ii)];
        buffer[ALIGN_IDX(blk, bxlen, tid, ii)] =
            buffer[ALIGN_IDX(blk, bxlen, tid, min)];
        buffer[ALIGN_IDX(blk, bxlen, tid, min)] = t;
      }
    }
  }
}

// #define DEBUG

__global__ void mem_bank_1d_kernel(const float *in, float *out, int len,
                                   int limit, int boxsz) {
  unsigned int xstride = blockDim.x * gridDim.x;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ extern float buffer[];

#ifdef DEBUG
  printf("tid: %d/%d stride: %d limit: %d\n", tid, xstride, xstride, limit);
#endif

  for (unsigned int idx = tid; idx < limit; idx += xstride) {

    int blk = idx / 32;
    int bank = tid % 32;

#ifdef DEBUG
    printf("idk: %d bank: %d blk: %d\n", idx, bank, blk);
#endif

    // Populate banks
    if (tid % 32 == 0) {
      // int b_idx = 0;

      // Blocks to populate
      // int blkst = tid / 32;
      // int blkend = (tid + xstride) / 32;
      // for (int block = blkst; block < blkend; block++) {
      // int st = blk * 32;
#pragma unroll
      for (int i = 0; i < boxsz; i++) {
#pragma unroll
        for (int bank = 0; bank < 32; bank++) {
          int k = (32 * blk) + i + bank;
          buffer[ALIGN_IDX(blk, boxsz, bank, i)] = k < len ? in[k] : 10000.0;
        }
      }
      // }
    }
    __syncthreads();

#ifdef DEBUG
    if (tid == 0) {
      printf("Printing after populate\n");
      // int blkst = tid / 32;
      // int blkend = (tid + xstride) / 32;
      for (int block = 0; block < (limit / 32) + 1; block++) {
        // int st = blk * 32;
        for (int i = 0; i < boxsz; i++) {
          for (int bank = 0; bank < 32; bank++) {
            printf("%f ", buffer[ALIGN_IDX(block, boxsz, bank, i)]);
          }
          printf("\n");
        }
      }
      printf("\n");
    }

#endif

#ifdef DEBUG
    printf("thread: %d sorting block: %d bank: %d\n", idx, blk, bank);
#endif

    int min;
    int nth = boxsz / 2;
    // Generic sort algorithm, could be replace with any other algorithm
    for (int ii = 0; ii <= nth; ii++) {
      min = ii;
      // printf("%d %d\n", ALIGN_IDX(tid, 0), ALIGN_IDX(tid, 1));
      for (int jj = ii; jj < boxsz; jj++) {
        if (buffer[ALIGN_IDX(blk, boxsz, bank, jj)] <
            buffer[ALIGN_IDX(blk, boxsz, bank, min)]) {
          min = jj;
        }
      }

      float t = buffer[ALIGN_IDX(blk, boxsz, bank, ii)];
      buffer[ALIGN_IDX(blk, boxsz, bank, ii)] =
          buffer[ALIGN_IDX(blk, boxsz, bank, min)];
      buffer[ALIGN_IDX(blk, boxsz, bank, min)] = t;
    }

    float mid = buffer[ALIGN_IDX(blk, boxsz, bank, nth)];

    if (len % 2 == 0) {
      mid += buffer[ALIGN_IDX(blk, boxsz, bank, nth - 1)];
      mid /= 2.0;
    }

    // if (idx > 32) {
    //   printf("thread: %d mid: %f 0th %f 0idx %d \n", idx, mid,
    //          buffer[ALIGN_IDX(blk, boxsz, bank, nth)],
    //          ALIGN_IDX(blk, boxsz, bank, nth));
    // }

    out[idx] = mid;

    __syncthreads();

#ifdef DEBUG
    if (tid == 0) {
      // int blkst = tid / 32;
      // int blkend = (tid + xstride) / 32;
      for (int block = 0; block < (limit / 32) + 1; block++) {
        // int st = blk * 32;
        for (int i = 0; i < boxsz; i++) {
          for (int bank = 0; bank < 32; bank++) {
            printf("%f ", buffer[ALIGN_IDX(block, boxsz, bank, i)]);
          }
          printf("\n");
        }
      }
      printf("\n");
    }

#endif
  }
}

void run_kernel() {}

void print_usage() {
  printf("Usage ./population [Options]\n");
  printf("Options\n");
  printf("\t-l\t\tDimension of array\n");
  printf("\t-b\t\tDimension of box\n");
  printf("\t-v\t\tPrint box fill\n");
}

int main(int argc, char **argv) {
  int len = 0;
  int boxsz = 0;
  int visualize = 0;

  if (argc < 2) {
    print_usage();
    return 1;
  }

  int c;
  while ((c = getopt(argc, argv, "l:b:hv")) != -1) {
    switch (c) {
    case 'l':
      len = atoi(optarg);
      break;
    case 'b':
      boxsz = atoi(optarg);
      break;
    case 'v':
      visualize = 1;
      break;
    default:
      print_usage();
      return 1;
    }
  }

  size_t limit = len - boxsz;
  size_t size = (sizeof(float) * 32 * boxsz) * ((limit / boxsz) + 1);

  printf("dimension: %d blocksz: %d allocating %lu \n", len, boxsz, size);

  float *in, *out;
  cudaMallocManaged(&in, sizeof(float) * len);
  cudaMallocManaged(&out, sizeof(float) * limit);

  for (int i = 0; i < len; i++) {
    in[i] = (len - i) * 1.0;
  }

  mem_bank_1d_kernel<<<1, 64, size>>>(in, out, len, limit, boxsz);

  cudaDeviceSynchronize();

  for (int i = 0; i < limit; i++) {
    printf("%f ", out[i]);
  }
  printf("\n");

  cudaFree(in);
  cudaFree(out);
}
