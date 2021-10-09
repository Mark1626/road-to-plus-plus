#include <fmt/core.h>
#include <iostream>
#include <cmath>

static const int ITERATION_LIMIT = 1000;

float A[4][4] = {{10., -1., 2., 0.},
                 {-1., 11., -1., 3.},
                 {2., -1., 10., -1.},
                 {0.0, 3., -1., 8.}};

float b[] = {6., 25., -11., 15.};

const int N = 4;
void solution() {
  fmt::print("System:\n");
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {

      fmt::print("{}*x{}", A[j][i], i + 1);
      if (i != N - 1)
        fmt::print(" + ");
    }
    fmt::print("= {}\n", b[j]);
  }

  float Anew[N][N] = {0};
  float error = 0;
  for (int it_count = 0; it_count < ITERATION_LIMIT; it_count++) {
    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            Anew[j][i] = 0.25 * (A[j][i] + A[j][i-1] + A[j-1][i] + A[j+1][i]);
            error = fmax(error, fabs(Anew[j][i] - A[j][i]));
        }
    }

    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            A[j][i] = Anew[j][i];
        }
    }

    fmt::print("{}, {}\n", it_count, error);
  }

  fmt::print("Solution: ") fmt::print(x) error =
      np.dot(A, x) - b fmt::print("Error: ") fmt::print(error)
}
