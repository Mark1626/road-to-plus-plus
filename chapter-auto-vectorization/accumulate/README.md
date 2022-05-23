# Auto vectorization of accumulate case study

## Case-1

Surprisingly the following loop is not vectorized

```c
void fn_accumulate(float *a, float *b,  float *c, uint32_t N, uint32_t K) {
  for (uint32_t i = 0; i < N; i++) {
    c[K] += a[i] * b[i];
  }
}
```

The OpenMP SIMD contruct is able to vectorize it

```c
void fn_omp_simd_accumulate(float *a, float *b,  float *c, uint32_t N, uint32_t K) {
  float sum = 0;

  #pragma omp simd reduction(+:sum)
  for (uint32_t i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }
  c[K] = sum;
}
```

### Performance

------------------------------------------------------------------------------
Benchmark                           |         Time     |        CPU  | Iterations
------------------------------------|------------------|-------------|-----------
BM_Accumulate_Mul/262144            |     0.163 ms     |   0.163 ms  |       4300
BM_Accumulate_Mul/2097152           |      1.30 ms     |    1.30 ms  |        538
BM_Accumulate_Mul/16777216          |      11.0 ms     |    11.0 ms  |         64
BM_Accumulate_Mul/67108864          |      42.9 ms     |    42.9 ms  |         16
BM_Accumulate_Mul_Auto_Vec/262144   |     0.163 ms     |   0.163 ms  |       4301
BM_Accumulate_Mul_Auto_Vec/2097152  |      1.31 ms     |    1.31 ms  |        538
BM_Accumulate_Mul_Auto_Vec/16777216 |      11.0 ms     |    11.0 ms  |         63
BM_Accumulate_Mul_Auto_Vec/67108864 |      49.2 ms     |    49.2 ms  |         14
BM_Accumulate_Mul_SIMD/262144       |     0.020 ms     |   0.020 ms  |      34129
BM_Accumulate_Mul_SIMD/2097152      |     0.164 ms     |   0.164 ms  |       4266
BM_Accumulate_Mul_SIMD/16777216     |      3.62 ms     |    3.62 ms  |        193
BM_Accumulate_Mul_SIMD/67108864     |      16.2 ms     |    16.2 ms  |         43


