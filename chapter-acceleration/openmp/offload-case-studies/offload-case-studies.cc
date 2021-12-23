#include "case1.cc"
#include "case2.cc"
#include "case3.cc"

int main() {
  int N = 1000;

  // test_sum_int(N);
  // test_sum_float(N);
  // test_offload_struct(N);
  // test_offload_allocate(N);
  test_offload_allocate_malloc(N);
}