# Road to C++

This repo contains things I'm going to try out to learn C++

# Chapters

Below each chapter noting down the things I learnt in the process

- [Chapter 1 - Basics](./chapter-1/README.md)
  + [1.1]() Learnt a ton of new features available
  + [1.2]() Run Time Type Information
- [Chapter 2 - Game of Life](./chapter-2/README.md)
  - [2.1](./chapter-2/2.1/README.md) Basic GOL with setting up LLDB with VSCode
  - [2.2](./chapter-2/2.2/README.md) GOL with SSE, interesting thing I noticed is that the first execution was a bit slower compared to the others. CPU Cache??
- [Chapter 3 - SHA-1](./chapter-3/README.md)
  + Endian Conversion optimization with `OSByteOrder.h` for Mac, this is converted into `bswap`, checked it by inspecting the generated ASM
- [Chapter 4 - Perlin Noise](./chapter-4/README.md)
  + [4.1](./chapter-4/4.1/README.md) Generated Perlin Noise map for a 10^8 points, parallelized it with OpenMP
- [Chapter 5 - Fractals](./chapter-5/README.md)
  + [5.1](./chapter-5/5.1/README.md) Abelian sandpile model with SSE

- Paraphernalia
  + Basic spawning threads with pthread and <thread>
  + Basic forking process
  + SSE condition paraphernalia


- Talks
  + [Tuning C++: Benchmarks, and CPUs, and Compilers! Oh My!](https://www.youtube.com/watch?v=nXaxk27zwlk)
  + [LLVM Hybrid Datastructures](https://www.youtube.com/watch?v=vElZc6zSIXM)
  + [std::allocator](https://www.youtube.com/watch?v=LIb3L4vKZ7U)

## Tools and Things to Explore

- [perf](https://en.wikipedia.org/wiki/Perf_(Linux))
- [gprof]()
- [vtune](https://en.wikipedia.org/wiki/VTune)
- [Google Benchmark]()
- [Google Test]()
- [rr]()
- [lldb]()

## Guides

- [OpenMP Guide](https://bisqwit.iki.fi/story/howto/openmp/#Abstract)
- [CPP Tutor](https://github.com/banach-space/cpp-tutor)

## Rough brainstorm of ideas

- [] Go through [CPP Tutor](https://github.com/banach-space/cpp-tutor)
- [] Build a Game of Life which could work for large cells
- [] Segmented Sieve of Erathosthanes for 10**12!?
- [x] Understand debugging with `lldb`
- [] Try out `rr`
- [] Contribute to a OSS project using C++
- [] Write a toy compiler
