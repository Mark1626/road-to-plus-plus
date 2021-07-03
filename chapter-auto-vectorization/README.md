# Auto Vectorization

## Techniques

### LLVM Auto Vectorization

- Cost model for vectorization decides if vectorization can give a performance improvement

#### Clang report to check if vectorization was successfull

Adding optimization remarks to check if loop was vectorized

`-Rpass=loop-vectorize`

### OpenMP

```cpp
// Compile with output should be vectorized with OpenMP, we turn off 
// clang's auto vectorization
// clang++ -omain main.cc -O3 -fno-vectorize -fopenmp
const int s = 10;
void func(int *a, int *b, int N) {
    #pragma omp simd
    for (int i = 0; i < N; i++) {
    a[i] += s * b[i];
    }
}
```


### Common things to understand

#### restrict

```cpp
void mul(int *__restrict__ a, int *__restrict__ b) {
  for (int i = 0; i < 64; i++) {
    a[i] *= b[i];
  }
}
```

In the above example without the `__restrict__` the compiler has no way to know if the two pointers `a` and `b` will overlap.

`__restrict__` is to mention to the complier that for the scope of the pointer the target will only be accessed through that pointer

> **Note :** In clang with -O3 for the above example is vectorized but it's auto vectorization, but without `__restrict__` the compiler will generate two function a scalar and vector and use the corresponding one after checking for overlap. For further details read [Clang runtime checks of pointers](https://llvm.org/docs/Vectorizers.html#runtime-checks-of-pointers)

#### Case Study 1

In the below code vectorization is valid only when elements of `d` are distinct, I couldn't get clang and gcc to vectorize it. I was able to vectorize by adding the `omp simd` pragma

[Case Study 1 in Godbolt](https://godbolt.org/z/c68h7z6Ks)

```cpp
#pragma omp simd
for (int i = 0; i < N; i++) {
  int j = d[i];
  a[j] += s * b[i];
}
```

#### Case Study 2 - Functions

```cpp
float s;
#pragma omp simd
for (int i = 0; i < N; i++) {
  int j = d[i];
  a[j] += foo(s, &b[i], a[j]);
}
```

## Reference

- [LLVM Loop Vectorization](https://llvm.org/docs/Vectorizers.html#the-loop-vectorizer)
- [Open MP Auto Vectorization](https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-simd.html)
- [Open MP Vectorization Examples](https://hpac.cs.umu.se/teaching/pp-16/material/08.OpenMP-4.pdf)

## See Also

- [Auto Vectorization](https://mark1626.github.io/knowledge/languages/c-compiler/auto-vectorization.html)
- [GCC Open Vectorization Tasks](https://gcc.gnu.org/wiki/VectorizationTasks)

