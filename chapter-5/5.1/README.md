# Chapter 5.1: Abelian sandpile model

Intro to Streaming SIMD instructions. Program is a single threaded implementation to find the identity sandpile

## Usage

For final single image

- `make all WITH_SSE=1 SIZE=6`
- `./main > sandpile.ppm`

For animation by pipeing to video player

- `make all WITH_SSE=1 SIZE=6 ANIMATE=1`
- `./main| mpv --no-correct-pts --fps=60 -`

## Performance

### 2**8

#### Normal execution

- `make all SIZE=8`
- `./main > sandpile.ppm`

Normal execution

```sh
❯ time ./main > test.ppm
./main > test.ppm  6.92s user 0.00s system 95% cpu 7.262 total

❯ time ./main > test.ppm
./main > test.ppm  6.93s user 0.00s system 99% cpu 6.934 total

❯ time ./main > test.ppm
./main > test.ppm  6.92s user 0.00s system 99% cpu 6.927 total
```

`make all WITH_SSE=1 SIZE=8`

#### SSE

```sh
❯ time ./main > test-sse.ppm
./main > test-sse.ppm  0.47s user 0.00s system 99% cpu 0.473 total

❯ time ./main > test-sse.ppm
./main > test-sse.ppm  0.54s user 0.00s system 99% cpu 0.543 total

❯ time ./main > test-sse.ppm
./main > test-sse.ppm  0.59s user 0.00s system 99% cpu 0.598 total
```

I have no words

### 2**10

![2**10 Sandpile](./test-sse.ppm)

- `make all WITH_SSE=1 SIZE=10`
- `./main > sandpile.ppm`

Normal execution has no chance

#### SSE

```
❯ time ./main > test-sse.ppm
./main > test-sse.ppm  127.19s user 0.12s system 99% cpu 2:07.50 total
```

