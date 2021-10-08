// Based on this article
#include "ppm.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <cstring>
#include <cerrno>

using std::cout;
using std::endl;
using std::vector;

typedef uint8_t u8;
typedef vector<u8> Array;
typedef vector<Array> Matrix;

void hist_equal(const Matrix input, Matrix &output) {
  int N = input.size(); // Assuming square

  // Calc occurance of intensity
  // Histogram
  std::map<int, int> pxl_histogram;
  for (auto j = 0; j < N; ++j) {
    for (auto i = 0; i < N; ++i) {
      if (!pxl_histogram[input[j][i]]) {
        pxl_histogram[input[j][i]] = 0;
      }
      pxl_histogram[input[j][i]]++;
    }
  }

  // CDF
  std::map<int, int> pxl_cdf;
  auto prev = 0;
  pxl_cdf[0] = 0;
  for (auto it : pxl_histogram) {
    pxl_cdf[it.first] = pxl_cdf[prev] + it.second;
    prev = it.first;
  }

  std::map<int, int> pxl_hist_equ;
  for (auto it : pxl_cdf) {
    pxl_hist_equ[it.first] = round((((double)it.second - 1.0) / 63.0) * 255.0);
  }

  #ifdef DEBUG
  for (auto it : pxl_hist_equ) {
    std::cout << it.first << " " << pxl_histogram[it.first] << " " << pxl_cdf[it.first] << " " << it.second << "\n";
  }
  #endif

  for (auto j = 0; j < N; ++j) {
    for (auto i = 0; i < N; ++i) {
      output[j][i] = pxl_hist_equ[input[j][i]];
    }
  }
}

void serialize(const Matrix m, FILE* file) {
  int N = m.size();
  ppm img = ppm_init(N, N);
  for (auto j = 0; j < N; ++j) {
    for (auto i = 0; i < N; ++i) {
      ppm_set(&img, i, j, {.r = m[j][i], .g = m[j][i], .b = m[j][i]});
    }
  }
  ppm_serialize(&img, file);
}

int main() {

  Matrix input({
    {52,55,61,59,79,61,76,61},
    {62,59,55,104,94,85,59,71},
    {63,65,66,113,144,104,63,72},
    {64,70,70,126,154,109,71,69},
    {67,73,68,106,122,88,68,68},
    {68,79,60,70,77,66,58,75},
    {69,85,64,58,55,61,65,83},
    {70,87,69,68,65,73,78,90}
  });
  Matrix output({
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}
  });

  FILE* input_file = fopen("input.ppm", "w");
  if (input_file == NULL) {
    std::cout << "Error due to " << strerror(errno) << std::endl;
    return 1;
  }
  serialize(input, input_file);
  fclose(input_file);

  // Matrix input({{1, 2, 3}, {255, 255, 2}, {7, 8, 9}});
  // Matrix output({{0,0,0}, {0, 0, 0}, {0,0,0}});
  hist_equal(input, output);

  FILE* output_file = fopen("output.ppm", "w");
  if (output_file == NULL) {
    std::cout << "Error due to " << strerror(errno) << std::endl;
    return 1;
  }
  serialize(output, output_file);
  fclose(output_file);
  return 0;
}
