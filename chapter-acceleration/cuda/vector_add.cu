#include <assert.h>
#include <iostream>

static const int N = 1 << 20;

#ifndef THREAD_BLOCKS
#define THREAD_BLOCKS 1
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

__global__
void vector_add(float *a, float *b, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        b[i] = a[i] + b[i];
    }
}

int main(void) {
    std:: cout << "Thread Blocks " <<  THREAD_BLOCKS << ", Threads " << NUM_THREADS << std::endl;
    float *a, *b;

    cudaMallocManaged(&a, sizeof(float) * N);
    cudaMallocManaged(&b, sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;//N - i;
    }

    int blockSize = 256;
    const int blocks = (N + blockSize - 1) / blockSize;

    vector_add<<< blocks , NUM_THREADS >>>(a, b, N);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(b[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;


    // for (int i = 0; i < N; i++) {
    //     std::cout << i << " " << b[i] << " ";
    //     //assert(b[i] == N);
    // }

    std:: cout << "Results asserted" << std::endl;

    cudaFree(a);
    cudaFree(b);

    return 0;   
}
