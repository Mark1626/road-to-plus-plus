
# GPUs

[AMD inline assembly](https://github.com/RadeonOpenCompute/ROCm/issues/405)

# OpenCL

> The [C++ wrapper](https://www.khronos.org/registry/OpenCL/api/2.1/cl.hpp) wrapper can be used to make it easier for C++

## C++ Wrapper

### Kernel Functor

```cpp
// OpenCL 1.1
cl::KernelFunctor simple_add(cl::Kernel(program,"simple_add"),queue,cl::NullRange, cl::NDRange(10),cl::NullRange);
simple_add(buffer_A,buffer_B,buffer_C);
```

```cpp
// Without Functors
cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
kernel_add.setArg(0,buffer_A);
kernel_add.setArg(1,buffer_B);
kernel_add.setArg(2,buffer_C);
queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange);
queue.finish();
```

```cpp
//OpenCL 1.2
cl::make_kernel<cl::Buffer&> atomic_sum(cl::Kernel(program, "atomic_sum"));
cl::EnqueueArgs args(queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
atomic_sum(args, bufferSum).wait();
```

## Synchronization

- Barriers and memory fence

## Pipes

A pipe is a memory

## Examples online

- [Sierpinski Carpet](https://software.intel.com/content/www/us/en/develop/articles/sierpinski-carpet-in-opencl-20.html)
- 

## Extensions

[OpenCL Extensions](https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/EXTENSION.html) available in OpenCL 2.1

### cl_khr_fp64

Support for double floating point precision

[cl_khr_fp64](https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/cl_khr_fp64.html)

```cpp
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
```

## References and Reading

- [OpenCL Atomic Performance](https://simpleopencl.blogspot.com/2013/04/performance-of-atomics-atomics-in.html)

- [OpenCL Basics](https://sites.google.com/site/csc8820/opencl-basics/opencl-concepts)
- [OpenCL Reference](https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/)


# Tools and Resources

- [AMD ADL SDK](https://gpuopen.com/adl/)
- [AMD uprof](https://developer.amd.com/amd-uprof/)

- [GPU Open Libraries and SDKs](https://github.com/GPUOpen-LibrariesAndSDKs)
- [GPU Open Tools](https://github.com/GPUOpen-Tools)

# OpenACC / OpenMP

- reduction - Reduction operation
- vector_add - Adding two vectors
- collide - Find partial hash collision in xxtea

Major learning - Be cautious of the work getting scheduled, unlike OMP in CPU there is no cancel

## References and Reading

- [OpenMP and GPUs](https://www.psc.edu/wp-content/uploads/2021/06/OpenMP-and-GPUs.pdf)
- [OpenACC Programming Guide](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0_0.pdf)
- [OpenMP GPUs](https://on-demand.gputechconf.com/gtc/2016/presentation/s6510-jeff-larkin-targeting-gpus-openmp.pdf)
