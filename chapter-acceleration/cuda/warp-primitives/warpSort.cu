// nvcc -o warpSort warpSort.cu --extended-lambda
#include <iostream>
#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

// Reference https://tschmidt23.github.io/cse599i/CSE%20599%20I%20Accelerated%20Computing%20-%20Programming%20GPUs%20Lecture%2018.pdf

__device__ __forceinline__ unsigned int bfe(unsigned int source, unsigned int bitIdx) {
  unsigned int bit;
  // ASM : output operands : input operands
  asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(bit) : "r"((unsigned int) source), "r"(bitIdx), "r"(1));
  return bit;
}


__device__ __forceinline__ float comparator(const float val, const int stride, const int dir) {
  const float other = __shfl_xor_sync(0xffffffff, val, stride);
  return val < other == dir ? other : val;
}

__device__ __forceinline__ void warpSort(float &val) {
  uint laneId = threadIdx.x % warpSize;

  val = comparator(val, 1, bfe(laneId, 1) ^ bfe(laneId, 0)); // A, sorted sequences of length 2
  val = comparator(val, 2, bfe(laneId, 2) ^ bfe(laneId, 1)); // B
  val = comparator(val, 1, bfe(laneId, 2) ^ bfe(laneId, 0)); // C, sorted sequences of length 4
  val = comparator(val, 4, bfe(laneId, 3) ^ bfe(laneId, 2)); // D
  val = comparator(val, 2, bfe(laneId, 3) ^ bfe(laneId, 1)); // E
  val = comparator(val, 1, bfe(laneId, 3) ^ bfe(laneId, 0)); // F, sorted sequences of length 8
  val = comparator(val, 8, bfe(laneId, 4) ^ bfe(laneId, 3)); // G
  val = comparator(val, 4, bfe(laneId, 4) ^ bfe(laneId, 2)); // H
  val = comparator(val, 2, bfe(laneId, 4) ^ bfe(laneId, 1)); // I
  val = comparator(val, 1, bfe(laneId, 4) ^ bfe(laneId, 0)); // J, sorted sequences of length 16
  val = comparator(val, 16, bfe(laneId, 4)); // K
  val = comparator(val, 8, bfe(laneId, 3)); // L
  val = comparator(val, 4, bfe(laneId, 2)); // M
  val = comparator(val, 2, bfe(laneId, 1)); // N
  val = comparator(val, 1, bfe(laneId, 0)); // O, sorted sequences of length 32
}

__global__ void warpSort(float *in, int N) {
  uint idx = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (idx < N) {
    warpSort(in[idx]);
  }
}

int main() {
  int N = 1024;

  thrust::host_vector<float> arr(N);
  thrust::device_vector<float> arr_d(N);
  thrust::default_random_engine rng(1337);
  thrust::uniform_int_distribution<int> dist;

  thrust::generate(arr.begin(), arr.end(), [&] { return dist(rng) % 100; });
  
  arr_d = arr;

  warpSort<<<4, 256>>>(thrust::raw_pointer_cast(arr_d.data()), N);

  arr = arr_d;

  std::cout << std::setprecision(2);
  for (int i = 0; i < N; i++) {
    std::cout << arr[i] << " ";
    if ((i+1) % 32 == 0) {
      std::cout << std::endl;
    }
  }

}
