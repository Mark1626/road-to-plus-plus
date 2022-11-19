#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>
#include <vector>

#ifdef __HIP_PLATFORM_HCC__
#define SHFL_DOWN(val, offset) __shfl_down(val, offset)
#else
#define SHFL_DOWN(val, offset) __shfl_down_sync(0xffffffff, val, offset)
#endif

inline void checkGPU(cudaError_t result, std::string file, int const line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error at %s:%d : %s\n", file.c_str(), line,
            cudaGetErrorString(result));
    exit(EXIT_FAILURE);
  }
}

#define checkGPUErrors(val) checkGPU(val, __FILE__, __LINE__)

inline void __getLastGPUError(const char *errorMessage, const char *file,
                              const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    std::cerr << file << "(" << line << ")"
              << " : getLastCudaError() CUDA error : " << errorMessage << " : "
              << (int)err << " " << cudaGetErrorString(err) << "." << std::endl;
    exit(EXIT_FAILURE);
  }
}


#define getLastGPUError(msg) __getLastGPUError(msg, __FILE__, __LINE__)


template <typename T> __inline__ __device__ T wrap_reduce_sum(T val) {
  for (uint i = warpSize / 2; i >= 1; i >>= 1)
    val += SHFL_DOWN(val, i);

  return val;
}

template <typename T> __inline__ __device__ T thread_block_reduce(T val) {
  static __shared__ T shared[64];
  uint tid = threadIdx.x;

  uint lane = tid % warpSize;
  uint warpid = tid / warpSize;

  val = wrap_reduce_sum(val);

  // Result of reduction in lane 0
  if (lane == 0) {
    shared[warpid] = val;
  }

  __syncthreads();

  val = (tid < blockDim.x / warpSize) ? shared[lane] : 0;

  if (warpid == 0) {
    val = wrap_reduce_sum(val);
  }

  return val;
}

template <typename T> __global__ void grid_reduce(T *in, T *out, int N) {
  T sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = thread_block_reduce(sum);
  if (threadIdx.x == 0) {
    atomicAdd(out, sum);
  }
}

template <typename T>
double sum_reduce_cuda(const T *arr, const size_t arrSize, T &sum) {
  const int blocks = 64;
  T out;

  cudaStream_t stream;
  cudaEvent_t start, stop;

  checkGPUErrors(cudaStreamCreate(&stream));
  checkGPUErrors(cudaEventCreate(&start));
  checkGPUErrors(cudaEventCreate(&stop));

  T *arr_d;
  T *out_d;
  checkGPUErrors(cudaMalloc(&arr_d, sizeof(T) * arrSize));
  checkGPUErrors(cudaMalloc(&out_d, sizeof(T) * 1));

  checkGPUErrors(cudaMemcpyAsync(arr_d, arr, sizeof(T) * arrSize,
                                 cudaMemcpyHostToDevice, stream));

  checkGPUErrors(cudaEventRecord(start, stream));

  grid_reduce<<<blocks, 256, 0, stream>>>(arr_d, out_d, arrSize);

  checkGPUErrors(cudaMemcpyAsync(&out, out_d, sizeof(T) * 1,
                                 cudaMemcpyDeviceToHost, stream));

  checkGPUErrors(cudaStreamSynchronize(stream));
  checkGPUErrors(cudaEventRecord(stop, stream));
  checkGPUErrors(cudaEventSynchronize(stop));

  float elapsedTime;
  checkGPUErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  getLastGPUError("Error running kernel");

  checkGPUErrors(cudaFree(arr_d));
  checkGPUErrors(cudaFree(out_d));

  sum = out;

  return elapsedTime;
}

int main() {
  int N = 1024;
  std::vector<float> x(N);
  thrust::sequence(x.begin(), x.end());
  float sum = 0.0f;

  sum_reduce_cuda<float>(x.data(), N, sum);

  std::cout << "Sum: " << sum << std::endl;
}
