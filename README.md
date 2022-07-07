# Road to C++ and High Performance Computing

This repo contains things I'm going to try out to learn C++ and High Performance Computing

A lot of these are content I gathered from various places of the internet. Raise an issue if I missed attributing any external content

# Chapters

Below each chapter noting down the things I learnt in the process

- [Chapter 1 - Basics](./chapter-1/)
  + 1.1 Learnt a ton of new features available
  + 1.2 Run Time Type Information
  + 1.3 [Constructors](./chapter-1/ctor.cpp)
  + 1.3 [Destructor order](./chapter-1/destruct.cc)
- [Chapter 2 - Game of Life](./chapter-2/)
  - [2.1](./chapter-2/2.1/) Basic GOL with setting up LLDB with VSCode
  - [2.2](./chapter-2/2.2/) GOL with SSE, interesting thing I noticed is that the first execution was a bit slower compared to the others.
- [Chapter 3 - SHA-1](./chapter-3/)
  + [3.1](./chapter-3/3.1/)Endian Conversion optimization with `OSByteOrder.h` for Mac, this is converted into `bswap`, checked it by inspecting the generated ASM
  + [3.2](./chapter-3/3.2/)Image Convolution
  + [3.3](./chapter-3/3.3/)Image Histogram Equalization
  + [3.4](./chapter-3/3.4/) Discrete Fourier Transform
- [Chapter 4 - Perlin Noise](./chapter-4/)
  + [4.1](./chapter-4/4.1/) Generated Perlin Noise map for a 10^8 points, parallelized it with OpenMP
  + [4.2](./chapter-4/4.2/)OpenMP case studies
- [Chapter 5 - Fractals](./chapter-5/)
  + [5.1](./chapter-5/5.1/) Abelian sandpile model with SSE
  + [5.2](./chapter-5/5.2/) Mandelbrot set with OpenMP
  + [5.3](./chapter-5/5.3/) Abelian sandpile model with auto vectorization for non square grids
- [Chapter Acceleration](./chapter-acceleration/)
  + OpenCL
    - **exp-1** Get available device info
    - **exp-2** Vector addition in kernel
    - **exp-3** Game of Life
    - **exp-4** Atomics
    - **exp-5** Abelian Sandpile(not the best example)
  + Cuda
    - **hello** Hello World with Cuda
    - **vector_add** Vector addition in device
  + OpenMP / OpenACC
    - **collide_gpu** xxtea middle block collision on GPU, based on [this gist](https://gist.github.com/skeeto/20d0768222af9e7fe6ec0a2d78726d1a)
    - **reduction** reduction operation on GPU
    - **vector_add** - Adding two vectors
- [Chapter CMake](./chapter-cmake/)
- [Chapter Flex Bison](./chapter-flex-bison/)
- [Chapter Multi Process](./chapter-multi-process/)
- [Chapter Auto Vectorization](./chapter-auto-vectorization/)
- [Chapter Profiling](./chapter-profiling/)
- [Chapter Benchmarks](./chapter-benchmark/)
- [Chaper CASA](./chapter-casa/)

- Paraphernalia
  + Basic spawning threads with pthread and `<thread>`
  + Basic forking process
  + SSE condition paraphernalia
  + Opening Ini file


## Talks
  + [Tuning C++: Benchmarks, and CPUs, and Compilers! Oh My!](https://www.youtube.com/watch?v=nXaxk27zwlk)
  + [LLVM Hybrid Datastructures](https://www.youtube.com/watch?v=vElZc6zSIXM)
  + [std::allocator](https://www.youtube.com/watch?v=LIb3L4vKZ7U)

- Reads
  + [Undefined Behaviour](https://blog.llvm.org/2011/05/what-every-c-programmer-should-know.html)
  + [Hybrid Parallel Programming](https://openmp.org/wp-content/uploads/HybridPP_Slides.pdf)

## Tools and Things to Explore

> Links are to respective documentations

- [perf](https://perf.wiki.kernel.org/index.php/Main_Page)
- [gprof](https://ftp.gnu.org/old-gnu/Manuals/gprof-2.9.1/html_mono/gprof.html)
- [valgrind](https://www.valgrind.org/docs/manual/quick-start.html)
- [vtune](https://en.wikipedia.org/wiki/VTune)
- [OpenMP](https://www.openmp.org/wp-content/uploads/OpenMPRefCard-5.1-web.pdf)
- [OpenMPI](https://www.open-mpi.org/doc/current/)
- [Google Benchmark](https://github.com/google/benchmark)
- [Google Test](https://github.com/google/googletest)
- [rr](https://github.com/rr-debugger/rr)
- [lldb](https://lldb.llvm.org/use/tutorial.html)

## Topics

- Intrinsics
- Stream Processing
- Domain Decomposition

## Papers

- [Selected Papers in ACM SIGPLAN 1979-1999](https://www.cs.utexas.edu/users/mckinley/20-years.html)
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](http://pages.cs.wisc.edu/~david/courses/cs552/S12/handouts/goldberg-floating-point.pdf)
- [What Every Programmer Should Know About Memory](https://people.freebsd.org//~lstewart/articles/cpumemory.pdf)
- [Shortlist of Top 10 Algorithms and links to their papers](https://people.math.sc.edu/Burkardt/classes/tta_2015/algorithms.html)

## Books

- [Introduction to High Performance Scientific Computing](https://pages.tacc.utexas.edu/~eijkhout/istc/html/index.html)
- [Computer Systems: A Programmer's Perspective]()
- [Linux Kernel in a Nutshell](http://www.kroah.com/lkn/)
- [Linux Device Drivers](https://lwn.net/Kernel/LDD3/)
- [C Unleashed](https://github.com/eliben/c-unleashed-book-souce-code)

## Guides

- [OpenMP Guide](https://bisqwit.iki.fi/story/howto/openmp/#Abstract)
- [OpenMP Examples](https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf)
- [OpenMP 4.0 - 4.5](https://asc.llnl.gov/sites/asc/files/2020-09/2-24_scogland.pdf)
- [CPP Tutor](https://github.com/banach-space/cpp-tutor)
- [StreamProcessing Reading List](https://github.com/ShuhaoZhangTony/StreamProcessing_ReadingList)
- [Domain Decomposition](https://github.com/arielshao/Domain-Decomposition-)
- [Awesome C++](https://github.com/rigtorp/awesome-modern-cpp)
- [Advanced OpenMP](https://openmpcon.org/wp-content/uploads/openmpcon2017/Tutorial2-Advanced_OpenMP.pdf)
- [Introduction to HPC](https://andreask.cs.illinois.edu/Teaching/HPCFall2012/)

## Blogs

- [NullProgram by skeeto](https://nullprogram.com/) C, C++, Performance, intrinsics
- [Krister Walfridsson’s blog](https://kristerw.blogspot.com/) LLVM, GCC, Valgrind, nvptx
- [Fredrik Johansson Blog](https://fredrikj.net/blog/) scientific computing, python
- [John Burkardt's Blog](https://people.math.sc.edu/Burkardt/) hpc, mpi, openmp
- [Scott Meyers](https://aristeia.com/)
- [Frank Tetzel](https://tetzank.github.io/)
- [Wojciech Muła](http://0x80.pl/)

