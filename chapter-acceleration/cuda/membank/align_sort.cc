#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

void populate(float *buff, float *a, int len, int limit, int boxsz) {
  int idx = 0;
  for (int x = 0; x < (limit / 32) + 1; x++) {
    int st = x * 32;
    for (int i = st; i < st + boxsz; i++) {
      for (int bank = 0; bank < 32; bank++) {
        int k = i + bank;
        buff[idx++] = k < len ? a[k] : 0xfffff;
      }
    }
  }
}

#define ALIGN_IDX(blk, bxlen, tid, idx) ((32 * bxlen * blk) + (32 * idx) + tid)

void align_sort(float *buffer, int limit, int bxlen) {

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
        buffer[ALIGN_IDX(blk, bxlen, tid, ii)] = buffer[ALIGN_IDX(blk, bxlen, tid, min)];
        buffer[ALIGN_IDX(blk, bxlen, tid, min)] = t;
      }
    }
  }
}

void print_buffer(float *buff, int limit, int boxsz) {
  int idx = 0;
  printf("\n");
  printf("Buffer\n");
  printf("-------------------\n");
  for (int x = 0; x < (limit / 32) + 1; x++) {
    int st = x * 32;
    for (int i = st; i < st + boxsz; i++) {
      for (int bank = 0; bank < 32; bank++) {
        printf("%f ", buff[idx++]);
      }
      printf("\n");
    }
  }
}

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

  int limit = len - boxsz;

  float *buff = new float[32 * boxsz * ((limit / 32) + 1)];
  float *a = new float[len];

  for (int i = 0; i < len; i++) {
    // a[i] = i * 1.0f;
    a[i] = (len - i) * 1.0f;
  }

  populate(buff, a, len, limit, boxsz);

  print_buffer(buff, limit, boxsz);

  align_sort(buff, limit, boxsz);

  print_buffer(buff, limit, boxsz);

  delete[] a;
  delete[] buff;
}
