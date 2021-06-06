# Road to C++

This repo contains things I'm going to try out to learn C++

# Chapters

Below each chapter noting down the things I learnt in the process

- [Chapter 1 - Basics](./chapter-1/README.md)
  + Learnt a ton of new features available
- [Chapter 2 - Game of Life](./chapter-2/README.md)
  - 2.1 Basic GOL with setting up LLDB with VSCode
  - 2.2 GOL with SSE, interesting thing I noticed is that the first execution was a bit slower compared to the others. CPU Cache??
- [Chapter 3 - SHA-1](./chapter-3/README.md)
  + Endian Conversion optimization with `OSByteOrder.h` for Mac, this is converted into `bswap`, checked it by inspecting the generated ASM
- [Chapter 4 - Perlin Noise](./chapter-4/README.md)
  + 4.1 Generated Perlin Noise map for a 10^8 points, parallelized it with OpenMP
- [Chapter 5 - Fractals](./chapter-5/README.md)
  + 5.1 Abelian sandpile model with SSE

- Paraphernalia
  + Basic spawning threads with pthread and <thread>
  + Basic forking process

## Rough brainstorm of ideas

- [] Go through [CPP Tutor](https://github.com/banach-space/cpp-tutor)
- [] Build a Game of Life which could work for large cells
- [] Segmented Sieve of Erathosthanes for 10**12!?
- [x] Understand debugging with `lldb`
- [] Try out `rr`
- [] Contribute to a OSS project using C++
- [] Write a toy compiler
