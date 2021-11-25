#include "casacore/casa/aipstype.h"
#include <casacore/casa/Arrays.h>
#include <iostream>

int main() {
  int SIZE = 8;

  // Matrix
  casacore::Matrix<casacore::Float> matrixA(SIZE, SIZE);
  casacore::Matrix<casacore::Float> matrixC(SIZE, SIZE);

  for (auto i = 0; i < SIZE; i++) {
    for (auto j = 0; j < SIZE; j++) {
      if (i == SIZE - 1) {
        if (j == 0) {
          matrixA(i, j) = 1;
        } else {
          continue;
        }
      } else if (j == i || j == i+1) {
        matrixA(i, j) = 1;
      }
    }
  }

  casacore::Matrix<casacore::Float> matrixB(matrixA);

  // Sadly I can't do this
  // matrixB = (matrixA * matrixA);

  // Standard matrix multiply
  for (auto i = 0; i < SIZE; i++) {
    for (auto j = 0; j < SIZE; j++) {
      casacore::Float temp = 0.0f;
      for (auto k = 0; k < SIZE; k++) {
        temp += matrixA(i, k) * matrixB(k ,j);
      }
      matrixC(i, j) = temp;
    }
  }

  for (auto i = 0; i < SIZE; i++) {
    for (auto j = 0; j < SIZE; j++) {
      std::cout << matrixC(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;


}
