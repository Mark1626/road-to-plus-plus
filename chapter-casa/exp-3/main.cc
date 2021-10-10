#include "casacore/casa/Arrays/ArrayFwd.h"
#include "casacore/casa/aipstype.h"
#include <casacore/casa/Arrays.h>
#include <iostream>

int main() {
  int SIZE = 8;

  // Matrix
  casacore::Matrix<casacore::Float> matrixA(SIZE, SIZE);
  casacore::Matrix<casacore::Float> matrixB(SIZE, SIZE);

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

  matrixB = (matrixA * matrixA);

  for (auto i = 0; i < SIZE; i++) {
    for (auto j = 0; j < SIZE; j++) {
      std::cout << matrixB(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;


}
