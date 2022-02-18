- exp-1: Comparison between managed memory and normal transfer
- exp-2: Masked sliding window sum with 2x output transferback
- exp-3: Masked sliding window median attempt-1(this does not work for large window sizes)
- exp-4: Comparison of per thread malloc memory allocation and blockDim and gridDim
- exp-5: qselect vs selection sort for median on 1D
  + This code has occupancy 97%, yet the performance is less than a CPU implementation, due to memory bandwidth used
- exp-6: Shared memory for storing window in kernel then process it
- exp-7: Statically allocating a large array in a kernel
- exp-8: Sliding window median with static memory allocation

https://ppl.stanford.edu/papers/sc11-bauer.pdf
