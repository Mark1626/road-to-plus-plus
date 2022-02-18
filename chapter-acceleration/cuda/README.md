# Cuda

## Cuda Usage

Calling device kernels

```
<<< M, T >>>

M - thread blocks
T - thread block size
```

## Stride Loops

```cpp
__global__ void kernel(int n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(; tid < n; tid += stride) {
        // Some action
    }
}
```

This is done to ensure thread reuse and Scalability


## Contents

- hello: cuda Hello World
- vector_add: Vector add in GPU
- classes: Usage of classes with Cuda
- interop: Using a cuda compiled kernel from C++
- memory: Unified memory vs transfer



## References

- [Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
- [GPU Memory Oversubscription Performance](https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/)
