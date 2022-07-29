#pragma once
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define FILTER_NAN(x) ((x) == (x) ? (x) : 0)
#define BOXCAR_MIN_ITER 3
#define BOXCAR_MAX_ITER 6

void optimal_filter_size_dbl(const double sigma, size_t *filter_radius,
                             size_t *n_iter) {
  *n_iter = 0;
  *filter_radius = 0;
  double tmp = -1.0;
  size_t i;

  for (i = BOXCAR_MIN_ITER; i <= BOXCAR_MAX_ITER; ++i) {
    const double radius = sqrt((3.0 * sigma * sigma / i) + 0.25) - 0.5;
    const double diff = fabs(radius - floor(radius + 0.5));

    if (tmp < 0.0 || diff < tmp) {
      tmp = diff;
      *n_iter = i;
      *filter_radius = (size_t)(radius + 0.5);
    }
  }
  return;
}

void filter_boxcar_1d_flt(float *data, float *data_copy, const size_t size,
                          const size_t filter_radius) {
  // Define filter size
  const size_t filter_size = 2 * filter_radius + 1;
  const float inv_filter_size = 1.0 / filter_size;
  size_t i;

  // Make copy of data, taking care of NaN
  for (i = size; i--;)
    data_copy[filter_radius + i] = FILTER_NAN(data[i]);

  // Fill overlap regions with 0
  for (i = filter_radius; i--;)
    data_copy[i] = data_copy[size + filter_radius + i] = 0.0;

  // Apply boxcar filter to last data point
  data[size - 1] = 0.0;
  for (i = filter_size; i--;)
    data[size - 1] += data_copy[size + i - 1];
  data[size - 1] *= inv_filter_size;

  // Recursively apply boxcar filter to all previous data points
  for (i = size - 1; i--;)
    data[i] = data[i + 1] +
              (data_copy[i] - data_copy[filter_size + i]) * inv_filter_size;

  return;
}

#include <immintrin.h>
#include <xmmintrin.h>

// #define FILTER_NAN(x) ((x) == (x) ? (x) : 0)

static inline __m128 filter_nan_sse(__m128 data_v) {
  // Filter nan
  __m128 nan_mask = _mm_cmpord_ps(data_v, data_v);
  __m128 zero_v = _mm_setzero_ps();
  __m128 data_filtered_v = _mm_blendv_ps(zero_v, data_v, nan_mask);
  return data_filtered_v;
}

static inline __m256 filter_nan_avx(__m256 data_v) {
  // Filter nan
  __m256 nan_mask = _mm256_cmp_ps(data_v, data_v, _CMP_ORD_Q);
  __m256 zero_v = _mm256_setzero_ps();
  __m256 data_filtered_v = _mm256_blendv_ps(zero_v, data_v, nan_mask);
  return data_filtered_v;
}

void print_vector(__m128 vec) {
  float res[4];
  _mm_storeu_ps(res, vec);
  for (int i = 0; i < 4; i++) {
    printf("%.1f ", res[i]);
  }
  printf("\n");
}

void filter_simd_sse(float *data, const size_t size, const size_t stride,
                 const size_t filter_radius) {
  const size_t filter_size = 2 * filter_radius + 1;
  size_t i;

  const float inv_filter_size = 1.0 / filter_size;
  __m128 inv_filter_size_v = _mm_set1_ps(inv_filter_size);
  __m128 zero_v = _mm_setzero_ps();

  __m128 *data_copy =
      (__m128 *)malloc(sizeof(__m128) * (size + 2 * filter_radius));

  for (i = size; i--;)
    data_copy[filter_radius + i] = filter_nan_sse(_mm_loadu_ps(data + (stride * i)));

  for (i = filter_radius; i--;)
    data_copy[i] = data_copy[size + filter_radius + i] = zero_v;

  // Calculate last point
  __m128 last_pt = zero_v;
  for (i = filter_size; i--;) {
    last_pt = _mm_add_ps(last_pt, data_copy[size + i - 1]);
  }
  last_pt = _mm_mul_ps(last_pt, inv_filter_size_v);
  _mm_storeu_ps(data + (stride * (size - 1)), last_pt);

  __m128 next_pt = last_pt;
  for (int col = size - 1; col--;) {
    __m128 current_pt = _mm_sub_ps(data_copy[col], data_copy[filter_size + col]);
    current_pt = _mm_mul_ps(current_pt, inv_filter_size_v);
    current_pt = _mm_add_ps(next_pt, current_pt);

    _mm_storeu_ps(data + (stride * col), current_pt);

    next_pt = current_pt;
  }

  free(data_copy);

  return;
}

void filter_simd_avx(float *data, const size_t size, const size_t stride,
                 const size_t filter_radius) {
  const size_t filter_size = 2 * filter_radius + 1;
  size_t i;

  const float inv_filter_size = 1.0 / filter_size;
  __m256 inv_filter_size_v = _mm256_set1_ps(inv_filter_size);
  __m256 zero_v = _mm256_setzero_ps();

  __m256 *data_copy =
      (__m256 *)malloc(sizeof(__m256) * (size + 2 * filter_radius));

  for (i = size; i--;)
    data_copy[filter_radius + i] = filter_nan_avx(_mm256_loadu_ps(data + (stride * i)));

  for (i = filter_radius; i--;)
    data_copy[i] = data_copy[size + filter_radius + i] = zero_v;

  // Calculate last point
  __m256 last_pt = zero_v;
  for (i = filter_size; i--;) {
    last_pt = _mm256_add_ps(last_pt, data_copy[size + i - 1]);
  }
  last_pt = _mm256_mul_ps(last_pt, inv_filter_size_v);
  _mm256_storeu_ps(data + (stride * (size - 1)), last_pt);

  __m256 next_pt = last_pt;
  for (int col = size - 1; col--;) {
    __m256 current_pt = _mm256_sub_ps(data_copy[col], data_copy[filter_size + col]);
    current_pt = _mm256_mul_ps(current_pt, inv_filter_size_v);
    current_pt = _mm256_add_ps(next_pt, current_pt);

    _mm256_storeu_ps(data + (stride * col), current_pt);

    next_pt = current_pt;
  }

  free(data_copy);

  return;
}

