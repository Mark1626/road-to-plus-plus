# OpenMP offloading to GPU

The example were offloaded to a Nvidia 1650, using GCC

## Case Studies

1. vector_add

Add of two arrays and store it in a third array
  + make vector_add_gpu
  + ./vector_add_gpu
  + make vector_add_cpu
  + ./vector_add_cpu

----------------------------------------------------

2. Offloading Case Studies

  + Case-1: Reduction using `critical` construct, comparison between threads and offload for float and int
  + Case-2: Target offload struct to device
  + Case-3: Memory allocation in device offloading.
  + Case-4: Offload qselect to device(not optimal due to thread divergence)
  + Case-5: Offload `class` methods to device

----------------------------------------------------

3. XXTEA block middle collision on GPU

----------------------------------------------------

4. Significance of memory access pattern

https://www.olcf.ornl.gov/wp-content/uploads/2021/08/ITOpenMP_Day1.pdf
