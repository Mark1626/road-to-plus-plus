#include <cstdint>
#include <iostream>
#include <chrono>
#include <cstdio>
#include <ratio>

using namespace std;

void native() {
  std::uint32_t sum = 0;
  for (std::uint32_t i = 0; i < 1000000000; ++i) {
    sum += i % 1000;
  }
  printf("Sum %d\n", sum);
}

void with_mp() {
  std::uint32_t sum = 0;
  #pragma omp parallel for reduction(+ : sum)
  for (std::uint32_t i = 0; i < 1000000000; ++i) {
    sum += i % 1000;
  }
  printf("Sum %d\n", sum);
}

void native_nested() {
  std::uint32_t sum = 0;
  for (std::uint32_t i = 0; i < 1000; ++i) {
    for (std::uint32_t j = 0; j < 1000000; ++j) {
      uint32_t idx = i * 1000000 + j;
      sum += idx % 1000;
    }
  }
  printf("Sum %d\n", sum);
}

void with_mp_nested() {
  std::uint32_t sum = 0;
  #pragma omp parallel for reduction(+ : sum)
  for (std::uint32_t i = 0; i < 1000; ++i) {
    for (std::uint32_t j = 0; j < 1000000; ++j) {
      uint32_t idx = i * 1000000 + j;
      sum += idx % 1000;
    }
  }
  printf("Sum %d\n", sum);
}

int main() {
  auto start = chrono::steady_clock::now();
  native();
  auto end = chrono::steady_clock::now();
  auto diff = end - start;
  printf("Time taken %f ms\n", chrono::duration <double, milli> (diff).count());

  start = chrono::steady_clock::now();
  with_mp();
  end = chrono::steady_clock::now();
  diff = end - start;
  printf("Time taken parallel %f ms\n", chrono::duration <double, milli> (diff).count());

  start = chrono::steady_clock::now();
  native_nested();
  end = chrono::steady_clock::now();
  diff = end - start;
  printf("Time taken %f ms\n", chrono::duration <double, milli> (diff).count());

  start = chrono::steady_clock::now();
  with_mp_nested();
  end = chrono::steady_clock::now();
  diff = end - start;
  printf("Time taken parallel %f ms\n", chrono::duration <double, milli> (diff).count());
}
