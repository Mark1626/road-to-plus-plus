# Chapter 4.1 - Intro into parallelization with OpenMP

Start with a embarrsingly parallel task. Compute perlin noise for `10**4 * 10**4` points

## Building

```
make all
```

## Goals

- Start with parallezation with OpenMP

> **Note:** Mac requires `libomp` to be able to use openmp

### Raw OpenMP

#### main

```sh
❯ time ./main
./main  3.58s user 0.09s system 90% cpu 4.034 total

❯ time ./main
./main  3.60s user 0.09s system 99% cpu 3.697 total

❯ time ./main
./main  3.64s user 0.11s system 99% cpu 3.753 total
```

---

#### omain

```sh
❯ time ./omain
./omain  3.22s user 0.00s system 86% cpu 3.744 total

❯ time ./omain
./omain  3.24s user 0.00s system 99% cpu 3.243 total

❯ time ./omain
./omain  3.25s user 0.00s system 99% cpu 3.259 total
```

---

#### pmain

Parallelize loop

```sh
❯ time ./pmain
./pmain  8.55s user 0.18s system 1176% cpu 0.742 total

❯ time ./pmain
./pmain  8.24s user 0.17s system 1177% cpu 0.714 total

❯ time ./pmain
./pmain  8.30s user 0.18s system 1170% cpu 0.725 total
```
---

#### pomain

Parallelize loop with -03

```sh
❯ time ./pomain
./pomain  9.71s user 0.17s system 906% cpu 1.090 total

❯ time ./pomain
./pomain  9.85s user 0.17s system 1182% cpu 0.848 total

❯ time ./pomain
./pomain  9.75s user 0.18s system 1163% cpu 0.853 total
```

---


