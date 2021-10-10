#include "casacore/casa/Arrays/Vector.h"
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/aipstype.h>
#include <casacore/casa/Arrays.h>
#include <iostream>

int main() {
  const int SIZE = 8;

  // Array
  std::cout << std::endl << "Array" << std::endl;

  casacore::Array<casacore::Float> arr(casacore::IPosition(2, SIZE, SIZE));
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      arr(casacore::IPosition(2, i, j)) = i * SIZE + j;
    }
  }

  std::cout << arr.shape() << std::endl;

  // Or this can be printed like this
  // for (auto val : arr) {
  //   std::cout << val << " ";
  // }
  // std::cout << std::endl;

  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      std::cout << arr(casacore::IPosition(2, i, j)) << " ";
    }
    std::cout << std::endl;
  }

  // Vector
  std::cout << std::endl << "Vector" << std::endl;
  casacore::Vector<casacore::Float> vector(SIZE);
  for (int i = 0; i < SIZE; ++i) {
    vector(i) = i;
  }

  for (int i = 0; i < SIZE; ++i) {
    std::cout << vector(i) << " ";
  }
  std::cout << std::endl;

  std::cout << std::endl << "Matrix" << std::endl;
  // Matrix
  casacore::Matrix<casacore::Float> matrix(SIZE, SIZE);

  for (auto i = 0; i < SIZE; i++) {
    for (auto j = 0; j < SIZE; j++) {
      matrix(i, j) = i * SIZE + j;
    }
  }

  for (auto i = 0; i < SIZE; i++) {
    for (auto j = 0; j < SIZE; j++) {
      std::cout << matrix(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << std::endl << "Cube" << std::endl;
  // Cube
  casacore::Cube<casacore::Float> cube(SIZE, SIZE, SIZE);
  for (auto i = 0; i < SIZE; i++) {
    for (auto j = 0; j < SIZE; j++) {
      for (auto k = 0; k < SIZE; k++) {
        cube(i, j, k) = (i * SIZE * SIZE) + (j * SIZE) + k;
      }
    }
  }

  for (auto i = 0; i < SIZE; i++) {
    std::cout << "Slice i " << i << std::endl;
    for (auto j = 0; j < SIZE; j++) {
      for (auto k = 0; k < SIZE; k++) {
        std::cout << cube(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

}
