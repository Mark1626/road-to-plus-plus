# Road to C++ and High Performance Computing

This repo contains things I'm going to try out to learn C++.

A lot of these are content I gathered from various places of the internet. Raise an issue if I missed attributing any external content

# Chapters

Below each chapter noting down the things I learnt in the process

- [Chapter 1 - Basics](./chapter-1/)
  + 1.1 Learnt a ton of new features available
  + 1.2 Run Time Type Information
- [Chapter 2 - Game of Life](./chapter-2/)
  - [2.1](./chapter-2/2.1/README.md) Basic GOL with setting up LLDB with VSCode
  - [2.2](./chapter-2/2.2/README.md) GOL with SSE, interesting thing I noticed is that the first execution was a bit slower compared to the others. CPU Cache??
- [Chapter 3 - SHA-1](./chapter-3/)
  + Endian Conversion optimization with `OSByteOrder.h` for Mac, this is converted into `bswap`, checked it by inspecting the generated ASM
- [Chapter 4 - Perlin Noise](./chapter-4/)
  + [4.1](./chapter-4/4.1/README.md) Generated Perlin Noise map for a 10^8 points, parallelized it with OpenMP
- [Chapter 5 - Fractals](./chapter-5/)
  + [5.1](./chapter-5/5.1/README.md) Abelian sandpile model with SSE
- [Chapter CMake](./chapter-cmake/)
- [Chapter Flex Bison](./chapter-flex-bison/)
- [Chapter Multi Process](./chapter-multi-process/)
- [Chapter Auto Vectorization - WIP](./chapter-auto-vectorization/)

- Paraphernalia
  + Basic spawning threads with pthread and `<thread>`
  + Basic forking process
  + SSE condition paraphernalia
  + Open MPI


- Talks
  + [Tuning C++: Benchmarks, and CPUs, and Compilers! Oh My!](https://www.youtube.com/watch?v=nXaxk27zwlk)
  + [LLVM Hybrid Datastructures](https://www.youtube.com/watch?v=vElZc6zSIXM)
  + [std::allocator](https://www.youtube.com/watch?v=LIb3L4vKZ7U)

- Reads
  + [Undefined Behaviour](https://blog.llvm.org/2011/05/what-every-c-programmer-should-know.html)

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

## Books

- [Introduction to High Performance Scientific Computing](https://pages.tacc.utexas.edu/~eijkhout/istc/html/index.html)

## Guides

- [OpenMP Guide](https://bisqwit.iki.fi/story/howto/openmp/#Abstract)
- [CPP Tutor](https://github.com/banach-space/cpp-tutor)

## Blogs

- [NullProgram by skeeto](https://nullprogram.com/)

## Rough brainstorm of ideas

- [] Go through [CPP Tutor](https://github.com/banach-space/cpp-tutor)
- [x] Understand debugging with `lldb`
- [] Try out `rr`
- [] Contribute to a OSS project using C++
- [] Write a toy compiler
