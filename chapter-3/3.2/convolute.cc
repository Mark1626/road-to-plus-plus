#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

typedef vector<float> Array;
typedef vector<Array> Matrix;

void convolute1D(const Array input, const Array kernel,
                Array &output) {
  int input_size = input.size();
  int grid_size = input.size();

  for (int i = 0; i < input_size; ++i) {
    output[i] = 0;
    for (int j = 0; j < grid_size; ++j) {
      output[i] += input[i - j] * kernel[j];
    }
  }
}

void convolute2D(const Matrix input, const Matrix kernel, Matrix& output) {
  auto kcenterX = kernel.size() / 2;
  auto kcenterY = kernel.size() / 2;
  
  auto kernel_size = kernel.size();
  auto kernel_rows = kernel_size;
  auto kernel_cols = kernel_size;
  
  auto input_size = input.size();
  auto rows = input_size;
  auto cols = input_size;

  for (auto i = 0; i < rows; ++i) {
    for (auto j = 0; j < cols; ++j) {

      output[i][j] = 0;

      for (auto ki = 0; ki < kernel_rows; ++ki) {
        
        auto kernel_rows_idx = kernel_rows - 1 - ki;

        for (auto kj = 0; kj < kernel_cols; ++kj) {
          
          auto kernel_col_idx = kernel_cols - 1 - kj;

          auto ii = i + kcenterX - kernel_rows_idx;
          auto jj = j + kcenterY - kernel_col_idx;

          if (ii >= 0 && ii < rows && jj >= 0 && jj < cols) {
            output[i][j] += input[ii][jj] * kernel[kernel_rows_idx][kernel_col_idx];
          }
        }
      }
    }
  }
}

int main() {
  Array input({1.0f, 4.0f, 5.0f, 6.0f, 5.0f, 7.0f, 8.0f});
  Array kernel({0.2, 0.6, 0.2});
  Array output(7, 0);

  convolute1D(input, kernel, output);

  cout << "1D convolve input ";
  for (auto a : input) {
    cout << a << " ";
  }
  cout << endl;

  cout << "1D convolve output ";
  for (auto a : output) {
    cout << a << " ";
  }
  cout << endl << endl;

  Matrix input2D({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Matrix kernel2D({{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}});
  Matrix output2D({{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});

  convolute2D(input2D, kernel2D, output2D);

  cout << "2D convolve input " << endl;
  for (auto row : input2D) {
    for (auto val : row) {
      cout << val << " ";
    }
    cout << endl;
  }
  cout << endl;

  cout << "2D convolve output " << endl;
  for (auto row : output2D) {
    for (auto val : row) {
      cout << val << " ";
    }
    cout << endl;
  }
  cout << endl;
}
