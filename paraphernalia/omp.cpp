#include <cstddef>
#include <iostream>
#include <valarray>

int32_t find_max_slice(std::valarray<int32_t> slice, bool reverse = false) {
  int32_t max_sum = -0xffffff;
  int32_t sum = 0;

  int start = reverse ? slice.size() - 1 : 0;
  int end = reverse ? 0 : slice.size();

  for (int idx = start; reverse ? idx >= end : idx < end; reverse ? idx-- : idx++) {
    #ifdef DEBUG
    std::cout << idx << " : " << sum+slice[idx] << " : " << max_sum << "\n";
    #endif
    sum = std::max(-0xffffff, sum + slice[idx]);
    max_sum = std::max(max_sum, sum);
  }

  return max_sum;
}

int main() {
  // clang-format off
  std::valarray<int32_t> arr = {
    -2, 5, 3, 2, 
    9, -6, 5, 1,
    3, 2, 7, 3,
    -1, 8, -4, 8,
    };
  
  int max_sum = 0;
  // clang-format off
  #pragma omp parallel shared(max_sum)
  {
    // clang-format on

    // Row
    // clang-format off
    #pragma omp for nowait
    // clang-format on
    for (int i = 0; i < 4; i++) {
      std::valarray<int> slice = arr[std::slice(i * 4, 4, 1)];
      int segment_max_sum = 0;

      int slice_forward_sum = find_max_slice(slice);
      segment_max_sum = slice_forward_sum > segment_max_sum ? slice_forward_sum
                                                            : segment_max_sum;

      int slice_reverse_sum = find_max_slice(slice, true);
      segment_max_sum = slice_reverse_sum > segment_max_sum ? slice_reverse_sum
                                                            : segment_max_sum;

      if (segment_max_sum > max_sum) {
        // clang-format off
        #pragma omp critical
        {
          max_sum = segment_max_sum;
        }
        // clang-format on
      }
    }

    // Column
    // clang-format off
    #pragma omp for nowait
    // clang-format on
    for (int i = 0; i < 4; i++) {
      std::valarray<int> slice = arr[std::slice(i, 4, 4)];
      int segment_max_sum = 0;

      int slice_forward_sum = find_max_slice(slice);
      segment_max_sum = slice_forward_sum > segment_max_sum ? slice_forward_sum
                                                            : segment_max_sum;

      int slice_reverse_sum = find_max_slice(slice, true);
      segment_max_sum = slice_reverse_sum > segment_max_sum ? slice_reverse_sum
                                                            : segment_max_sum;

      if (segment_max_sum > max_sum) {
        // clang-format off
        #pragma omp critical
        {
          max_sum = segment_max_sum;
        }
        // clang-format on
      }
    }
  }

  std::cout << "Max Sum " << max_sum << "\n";
}
