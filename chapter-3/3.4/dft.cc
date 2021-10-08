#include "ppm.h"
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <numbers>
#include <vector>

using std::cout;
using std::endl;

typedef std::complex<float> Complex;
typedef std::vector<float> Array;

template <class T> class Matrix {
public:
  std::vector<T> val;
  int m;
  int n;
  Matrix() : m(0), n(0) {};
  Matrix(int m, int n) : m(m), n(n), val(m * n, 0){};
  T &operator()(int x, int y) { return val[y * m + x]; }
};
// typedef std::vector<Array> Matrix;

typedef std::vector<Complex> CArray;
typedef Matrix<Complex> CMatrix;

void dft2d(CMatrix input, CMatrix &output) {
  int N = input.n;
  int M = input.m;
  for (int k = 0; k < N; k++) {
    for (int l = 0; l < M; l++) {
      for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
          Complex t = std::polar<float>(
              1.0, (-2 * std::numbers::pi) *
                       (((float)k * n / N) + ((float)l * m / M)));
          output(k, l) += input(n, m) * t;
        }
      }
    }
  }
}

void dft(const Array input, CArray &output) {
  int N = input.size();
  for (int k = 0; k < N; k++) {
    for (int n = 0; n < N; n++) {
      Complex t = std::polar<float>(1.0, (-2 * std::numbers::pi * k * n) / N);
      output[k] += input[n] * t;
    }
  }
}

void verify1d(Array input) {
  int N = input.size();
  fftwf_complex *out;
  fftwf_plan p;
  out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N);

  p = fftwf_plan_dft_r2c_1d(N, input.data(), out, FFTW_ESTIMATE);

  fftwf_execute(p);

  for (int i = 0; i < N; i++) {
    cout << "(" << out[i][0] << "," << out[i][1] << ") ";
  }
  cout << endl;

  fftwf_destroy_plan(p);
  fftwf_free(out);
}

void test_1d() {
  Array input = {1, 1, 0, 0};
  CArray output(input.size(), 0);
  dft(input, output);

  cout << "Input\n";
  for (float val : input)
    cout << val << " ";
  cout << "\n";

  cout << "DFT\n";
  for (Complex val : output)
    cout << val << " ";
  cout << "\n";

  cout << "Verifying against FFTW3\n";
  verify1d(input);
}

// void verify2d(Matrix<float> input) {
//   int N = input.n;
//   int M = input.m;
//   fftwf_complex *out;
//   fftwf_plan p;
//   float *in = input.val.data();
//   out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N);

//   p = fftwf_plan_dft_r2c_2d(N, M, in, out, FFTW_ESTIMATE);

//   fftwf_execute(p);

//   for (int i = 0; i < M; i++) {
//     for (int j = 0; j < N; j++) {
//       cout << "(" << out[i * N + j][0] << "," << out[i * N + j][1] << ") ";
//     }
//     cout << endl;
//   }

//   fftwf_destroy_plan(p);
//   fftwf_free(out);
// }

// void test_2d() {
//   FILE *file = fopen("./input.ppm", "r");
//   // Matrix<float> mat;
//   if (file != NULL) {
//     ppm *img = ppm_parse(file);

//     printf("w: %d h: %d size: %lu\n", img->width, img->height,
//            sizeof(img->buff));
//     unsigned char *ptr = img->buff;
//     for (int w = 0; w < img->width; w++) {
//       for (int h = 0; h < img->height; h++) {
//         printf("%d ", *ptr);
//         ptr += 3;
//       }
//       printf("\n");
//     }

//     fclose(file);
//     free(img->buff);
//     free(img);
//   } else {
//     printf("Error occured %s\n", strerror(errno));
//   }
// }

int main() {
  test_1d();
  // test_2d();
  // std::fstream input("input.ppm", std::ios::in);
  return 0;
}
