# Chapter 2.2: Game of Life in SSE


> **Editor's Note:** When this was implemented I accidentally computed the normal iteration in the 
SSE variation along with the SSE iteration of GOL, resulting in me being puzzled on why and had me dig 
for hours into the ASM. And finally finding that the SSE variation is magnitudes faster is amazing

## Performance

### 2**9

```sh
❯ time ./main > test.ppm
./main > test.ppm  2.51s user 0.00s system 90% cpu 2.782 total

❯ time ./main > test.ppm
./main > test.ppm  2.52s user 0.00s system 99% cpu 2.524 total

❯ time ./main > test.ppm
./main > test.ppm  2.62s user 0.01s system 99% cpu 2.630 total
```

#### With SSE

> **Note :** Interesting thing is that the first execution was a bit slow. CPU Cache??

```
❯ time ./main > gol.ppm
./main > gol.ppm  0.09s user 0.00s system 20% cpu 0.459 total

❯ time ./main > gol.ppm
./main > gol.ppm  0.09s user 0.00s system 97% cpu 0.097 total

❯ time ./main > gol.ppm
./main > gol.ppm  0.10s user 0.00s system 96% cpu 0.106 total

❯ time ./main > gol.ppm
./main > gol.ppm  0.10s user 0.00s system 97% cpu 0.105 total
```

### 2**10

```
❯ time ./main > test.ppm
./main > test.ppm  9.98s user 0.01s system 96% cpu 10.372 total

❯ time ./main > test.ppm
./main > test.ppm  9.96s user 0.01s system 99% cpu 9.968 total

❯ time ./main > test.ppm
./main > test.ppm  10.04s user 0.01s system 99% cpu 10.056 total
```

#### With SSE

```
❯ time ./main > gol.ppm
./main > gol.ppm  0.31s user 0.01s system 48% cpu 0.666 total

❯ time ./main > gol.ppm
./main > gol.ppm  0.32s user 0.01s system 99% cpu 0.328 total

❯ time ./main > gol.ppm
./main > gol.ppm  0.34s user 0.01s system 99% cpu 0.353 total
```

#### Threading with Open MP
```
❯ time ./main > gol.ppm
./main > gol.ppm  36.93s user 0.08s system 1069% cpu 3.460 total

❯ time ./main > gol.ppm
./main > gol.ppm  37.02s user 0.09s system 1186% cpu 3.129 total

❯ time ./main > gol.ppm
./main > gol.ppm  38.26s user 0.12s system 1184% cpu 3.240 total
```

