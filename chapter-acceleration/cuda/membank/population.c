/*
  Program to visualize the population of banks for window sizes
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void stat_mem_bank_1d(int len, int boxsz) {
  int limit = len - boxsz;
  unsigned int block_volume = 32 * boxsz;
  unsigned int total_volume = block_volume * ((limit / boxsz) + 1);

  float size = (4.0 * total_volume) / (1024.0);

  printf("Total size used: %f \n", size);
  printf("Percentage of array possible to fit within target shared buffer size "
         "48k: %f \n",
         48.0 / size);
}

void visualize_mem_bank_1d(int len, int boxsz) {
  int limit = len - boxsz;
  for (int bank = 0; bank < 32; bank++) {
    printf("B%2d ", bank);
  }
  printf("\n");

  for (int x = 0; x < (limit / 32) + 1; x++) {
    int st = x * 32;
    for (int i = st; i < st + boxsz; i++) {
      for (int bank = 0; bank < 32; bank++) {
        int k = i + bank;
        k = k < len ? k : -1;
        printf("%3d ", k);
      }
      printf("\n");
    }
  }
}

void stat_mem_bank_2d(int len, int boxsz) {
  int limit = len - boxsz;
  unsigned int block_volume = 32 * boxsz * boxsz;
  unsigned int total_volume = block_volume * ((len / boxsz) + 1);

  float size = (4.0 * total_volume) / (1024.0);

  printf("Total size used: %f \n", size);
  printf("Percentage of array in possible within target shared buffer size "
         "48k: %f \n",
         48.0 / size);
}

void visualize_mem_bank_2d(int len, int boxsz) {
  int limit = len - boxsz;
  for (int bank = 0; bank < 32; bank++) {
    printf("B%2d ", bank);
  }
  printf("\n");

  for (int bank = 0; bank < 32; bank++) {
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

  printf("dimension: %d blocksz: %d\n", len, boxsz);

  if (visualize) {
    visualize_mem_bank_1d(len, boxsz);
  }

  stat_mem_bank_1d(len, boxsz);
}
