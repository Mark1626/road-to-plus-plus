#include <iostream>
#include <cassert>

static const int LIMIT = 1 << 26;

void add(int* __restrict__ a, int* __restrict__ b, int* __restrict__ c) {
  #pragma omp target map(to:a,b) map(tofrom:c)
  {
    #pragma omp teams distribute parallel for
    for (int i = 0; i < LIMIT; i++) {
      c[i] = a[i] + b[i];
    }
  }
}

int main() {

  int *a = new int[LIMIT];
  int *b = new int[LIMIT];
  int *c = new int[LIMIT];

  for (int i = 0; i < LIMIT; i++) {
    a[i] = i;
    b[i] = LIMIT - i;
    c[i] = 0;
  }

  add(a, b, c);

  delete []a;
  delete []b;

  for (int i = 0; i < LIMIT; i++) {
    // std::cout << c[i] << " ";
    assert(c[i] == LIMIT);
  }

  std::cout << "All elements asserted\n";

  delete []c;
}
