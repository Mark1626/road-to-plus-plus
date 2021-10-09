#include <stdio.h>

long long r = 1;

int main(void) {
  r = 10;
  #pragma omp target map(tofrom : r)
  {
    #pragma omp teams distribute parallel for reduction(+ : r)
    for (unsigned long long n = 0; n < 0x800000000ull; ++n) {
      r += n;
    }
  }
  printf("r=%llX\n", r);

  return 0;
}