#include <casacore/casa/Arrays.h>
#include <cassert>
#include <iostream>

namespace casa = casacore;

const int SIZE = 10;
const int WINDOW_SIZE = 2;

void sliding_medians(casa::Matrix<casa::Float> matrix) {
  casa::Array<casa::Float> medians = casa::slidingArrayMath(
      matrix, casa::IPosition(2, WINDOW_SIZE, WINDOW_SIZE),
      casa::MedianFunc<casa::Float>());

  std::cout << "Median: \n";
  for (auto median : medians) {
    std::cout << median << " ";
  }
  std::cout << std::endl;
}

void print_matrix(casa::Matrix<casa::Float> matrix) {
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      std::cout << matrix(casacore::IPosition(2, i, j)) << " ";
    }
    std::cout << std::endl;
  }
}

casa::Array<casa::Float>
test_sliding_masked_medians(casa::Matrix<casa::Float> matrix) {
  casa::MaskedArray<casa::Float> maskedMatrix(matrix, (matrix > 0.1f));
  casa::Array<casa::Float> maskedMedians = casa::slidingArrayMath(
      maskedMatrix, casa::IPosition(2, WINDOW_SIZE, WINDOW_SIZE),
      casa::MaskedMedianFunc<casa::Float>());

#ifdef DEBUG
  std::cout << "Masked Median: \n";
  for (auto median : maskedMedians) {
    std::cout << median << " ";
  }
  std::cout << std::endl;
#endif
  return maskedMedians;
}

class ModMedianFunc {
public:
  explicit ModMedianFunc(bool sorted = false, bool takeEvenMean = true)
      : itsSorted(sorted), itsTakeEvenMean(takeEvenMean) {}
  float operator()(const casa::MaskedArray<float> &arr) const {
    std::cout << arr << std::endl;
    return median(arr, itsSorted, itsTakeEvenMean);
  }

private:
  bool itsSorted;
  bool itsTakeEvenMean;
  bool itsInPlace;
};

casa::Array<float> optimSlidingArrayMath(const casa::MaskedArray<float> &array,
                                         const casa::IPosition &halfBoxSize,
                                         const ModMedianFunc &funcObj,
                                         bool fillEdge) {
  size_t ndim = array.ndim();
  const casa::IPosition &shape = array.shape();
  // Set full box size (-1) and resize/fill as needed.
  casa::IPosition hboxsz(2 * halfBoxSize);
  if (hboxsz.size() != array.ndim()) {
    size_t sz = hboxsz.size();
    hboxsz.resize(array.ndim());
    for (size_t i = sz; i < hboxsz.size(); ++i) {
      hboxsz[i] = 0;
    }
  }
  // Determine the output shape. See if anything has to be done.
  casa::IPosition resShape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    resShape[i] = shape[i] - hboxsz[i];
    if (resShape[i] <= 0) {
      if (!fillEdge) {
        return casa::Array<float>();
      }
      casa::Array<float> res(shape);
      res = float();
      return res;
    }
  }
  // Need to make shallow copy because operator() is non-const.
  casa::MaskedArray<float> arr(array);
  casa::Array<float> result(resShape);
  assert(result.contiguousStorage());
  float *res = result.data();
  // Loop through all data and assemble as needed.
  casa::IPosition blc(ndim, 0);
  casa::IPosition trc(hboxsz);
  casa::IPosition pos(ndim, 0);
  while (true) {
    *res++ = funcObj(arr(blc, trc));
    size_t ax;
    for (ax = 0; ax < ndim; ax++) {
      if (++pos[ax] < resShape[ax]) {
        blc[ax]++;
        trc[ax]++;
        std::cerr << "Updating BLC, TRC " << blc << trc
                  << " pos: " << pos << std::endl;
        break;
      }
      pos(ax) = 0;
      blc[ax] = 0;
      trc[ax] = hboxsz[ax];
      std::cerr << "Resetting BLC, TRC " << blc << trc
                << " pos: " << pos << std::endl;
    }
    if (ax == ndim) {
      break;
    }
  }
  if (!fillEdge) {
    return result;
  }
  casa::Array<float> fullResult(shape);
  fullResult = float();
  hboxsz /= 2;
  fullResult(hboxsz, resShape + hboxsz - 1).assign_conforming(result);
  return fullResult;
}

casa::Array<casa::Float> optim_call(casa::Matrix<casa::Float> &matrix) {
  casa::MaskedArray<casa::Float> maskedMatrix(matrix, (matrix > 0.1f));
  casa::Array<casa::Float> medians = optimSlidingArrayMath(
      maskedMatrix, casa::IPosition(2, WINDOW_SIZE, WINDOW_SIZE),
      ModMedianFunc(), true);

#ifdef DEBUG
  std::cout << "Median: \n";
  for (auto median : medians) {
    std::cout << median << " ";
  }
  std::cout << std::endl;
#endif

  return medians;
}

int main() {
  casa::IPosition shape(2, SIZE, SIZE);
  casa::Matrix<casa::Float> matrix(shape);
  double a = 5.0;

  for (auto &cell : matrix) {
    auto val = (double)std::rand() / (double)(RAND_MAX / a);
    cell = val;
  }

  // print_matrix(matrix);
  std::cout << matrix << std::endl;
  // sliding_medians(matrix);
  casa::Array<casa::Float> m1 = optim_call(matrix);
  // casa::Array<casa::Float> m2 = test_sliding_masked_medians(matrix);

  // std::cerr << m1.shape()[0] << m2.shape() << std::endl;

  // for (ssize_t i = 0; i < m1.shape()[0]; i++) {
  //   for (ssize_t j = 0; j < m1.shape()[1]; j++) {
  //     assert(fabs(m1(casa::IPosition(2, i, j)) - m2(casa::IPosition(2, i, j))
  //     <
  //                 0.0001));
  //   }
  // }
  // std::cout << "Assertion Done" << std::endl;

  return 0;
}
