#include "common.h"
#include "fits-helper.h"
#include "reference.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define TOLERANCE 0.01

void assert_array(float *expected, float *actual, size_t size_x, size_t size_y);

void filter_gauss_col_1(float *data, float *data_copy, float *data_row,
                        float *data_col, const size_t size_x,
                        const size_t size_y, const size_t n_iter,
                        const size_t filter_radius);
void filter_gauss_col_2(float *data, float *data_copy, float *data_row,
                        float *data_col, const size_t size_x,
                        const size_t size_y, const size_t n_iter,
                        const size_t filter_radius);

void filter_gauss_2d_sse(float *data, float *data_copy, float *data_row,
                         float *data_col, const size_t size_x,
                         const size_t size_y, const size_t n_iter,
                         const size_t filter_radius);

void filter_gauss_2d_avx(float *data, float *data_copy, float *data_row,
                         float *data_col, const size_t size_x,
                         const size_t size_y, const size_t n_iter,
                         const size_t filter_radius);

float *approach_1(const char *infilename, const char *outfilename);
float *approach_2(const char *infilename, const char *outfilename);
float *approach_3(const char *infilename, const char *outfilename);

void test_fits(const char *infilename, const char *outfilename1,
               const char *outfilename2, const char *outfilename3);

void compare_1iter();

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: imgauss <inimage> <outimage-1> <outimage-2> <outimage-3>\n");
    abort();
  }
  test_fits(argv[1], argv[2], argv[3], argv[4]);
}

void filter_gauss_col_1(float *data, float *data_copy, float *data_row,
                        float *data_col, const size_t size_x,
                        const size_t size_y, const size_t n_iter,
                        const size_t filter_radius) {
  const size_t size_xy = size_x * size_y;
  float *ptr = data + size_xy;
  float *ptr2;

  for (size_t x = size_x; x--;) {
    // Copy data into column array
    ptr = data + size_xy - size_x + x;
    ptr2 = data_copy + size_y;
    while (ptr2-- > data_copy) {
      *ptr2 = *ptr;
      ptr -= size_x;
    }

    // Apply filter
    for (size_t i = n_iter; i--;)
      filter_boxcar_1d_flt(data_copy, data_col, size_y, filter_radius);

    // Copy column array back into data array
    ptr = data + size_xy - size_x + x;
    ptr2 = data_copy + size_y;
    while (ptr2-- > data_copy) {
      *ptr = *ptr2;
      ptr -= size_x;
    }
  }

  return;
}

void filter_gauss_col_2(float *data, float *data_copy, float *data_row,
                        float *data_col, const size_t size_x,
                        const size_t size_y, const size_t n_iter,
                        const size_t filter_radius) {

  for (size_t x = 0; x < size_x; x += 4) {
    // Apply filter
    for (size_t i = n_iter; i--;) {
      filter_simd_sse(data + x, size_y, size_x, filter_radius);
    }
  }

  return;
}

void filter_gauss_2d_sse(float *data, float *data_copy, float *data_row,
                         float *data_col, const size_t size_x,
                         const size_t size_y, const size_t n_iter,
                         const size_t filter_radius) {
  // Set up a few variables
  const size_t size_xy = size_x * size_y;
  float *ptr = data + size_xy;

  // Run row filter (along x-axis)
  // This is straightforward, as the data are contiguous in x.
  while (ptr > data) {
    ptr -= size_x;
    for (size_t i = n_iter; i--;)
      filter_boxcar_1d_flt(ptr, data_row, size_x, filter_radius);
  }

  for (size_t x = 0; x < size_x; x += 4) {
    // Apply filter
    for (size_t i = n_iter; i--;) {
      filter_simd_sse(data + x, size_y, size_x, filter_radius);
    }
  }

  return;
}

void filter_gauss_2d_avx(float *data, float *data_copy, float *data_row,
                         float *data_col, const size_t size_x,
                         const size_t size_y, const size_t n_iter,
                         const size_t filter_radius) {
  // Set up a few variables
  const size_t size_xy = size_x * size_y;
  float *ptr = data + size_xy;

  // Run row filter (along x-axis)
  // This is straightforward, as the data are contiguous in x.
  while (ptr > data) {
    ptr -= size_x;
    for (size_t i = n_iter; i--;)
      filter_boxcar_1d_flt(ptr, data_row, size_x, filter_radius);
  }

  for (size_t x = 0; x < size_x; x += 8) {
    // Apply filter
    for (size_t i = n_iter; i--;) {
      filter_simd_avx(data + x, size_y, size_x, filter_radius);
    }
  }

  return;
}

void assert_array(float *expected, float *actual, size_t size_x,
                  size_t size_y) {
  for (size_t y = 0; y < size_y; y++) {
    for (size_t x = 0; x < size_x; x++) {
      float val = fabs(expected[y * size_x + x] - actual[y * size_x + x]);
      if (val > TOLERANCE) {
        printf("Error asserting point %ld %ld, expected: %f actual %f\n", x, y,
               expected[y * size_x + x], actual[y * size_x + x]);
        assert(val < TOLERANCE);
      }
    }
  }
}

void test_fits(const char *infilename, const char *outfilename1,
               const char *outfilename2, const char *outfilename3) {
  size_t size_x = 2048;
  size_t size_y = 2048;
  float *expected, *actual_sse, *actual_avx;
  actual_sse = expected = NULL;

  expected = approach_1(infilename, outfilename1);
  actual_sse = approach_2(infilename, outfilename2);
  actual_avx = approach_3(infilename, outfilename3);

  assert_array(expected, actual_sse, size_x, size_y);
  printf("Assertions passed SSE\n");

  assert_array(expected, actual_avx, size_x, size_y);
  printf("Assertions passed AVX\n");

  if (expected)
    free(expected);
  if (actual_sse)
    free(actual_sse);
  if (actual_avx)
    free(actual_avx);
}

float *approach_1(const char *infilename, const char *outfilename) {
  int status = 0;

  size_t n_iter;
  size_t filter_radius;
  double sigma = 3.5;
  optimal_filter_size_dbl(sigma, &filter_radius, &n_iter);
  printf("sigma: %f filter_radius: %ld niter: %ld\n", sigma, filter_radius,
         n_iter);

  fitsfile *fptr;
  fits_data_t fits;
  if (!fits_open_image(&fptr, infilename, READONLY, &status)) {
    status = extract_data_from_fits(fptr, &fits);
  } else {
    fits_report_error(stderr, status);
    abort();
  }

  float *data = fits.data;
  size_t size_x = fits.naxes[0];
  size_t size_y = fits.naxes[1];

  float *column = (float *)malloc(sizeof(float) * size_y);
  float *data_col =
      (float *)malloc(sizeof(float) * (size_y + 2 * filter_radius));
  float *data_row =
      (float *)malloc(sizeof(float) * (size_x + 2 * filter_radius));

  clock_t begin = clock();

  filter_gauss_2d_flt(data, column, data_row, data_col, size_x, size_y, n_iter,
                      filter_radius);

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time elapsed approach 1: %f s\n", time_spent);

  printf("Storing result in FITS\n");
  status = store_data_copying_fits_header(fptr, outfilename, &fits);

  // destroy_fits(&fits);
  fits_close_file(fptr, &status);

  if (status) {
    fits_report_error(stderr, status);
  }

  free(column);
  free(data_col);
  free(data_row);

  return fits.data;
}

float *approach_2(const char *infilename, const char *outfilename) {
  int status = 0;

  size_t n_iter;
  size_t filter_radius;
  double sigma = 3.5;
  optimal_filter_size_dbl(sigma, &filter_radius, &n_iter);
  printf("sigma: %f filter_radius: %ld niter: %ld\n", sigma, filter_radius,
         n_iter);

  fitsfile *fptr;
  fits_data_t fits;
  if (!fits_open_image(&fptr, infilename, READONLY, &status)) {
    status = extract_data_from_fits(fptr, &fits);
  } else {
    fits_report_error(stderr, status);
    abort();
  }

  float *data = fits.data;
  size_t size_x = fits.naxes[0];
  size_t size_y = fits.naxes[1];

  float *data_row =
      (float *)malloc(sizeof(float) * (size_x + 2 * filter_radius));

  clock_t begin = clock();

  filter_gauss_2d_sse(data, NULL, data_row, NULL, size_x, size_y, n_iter,
                     filter_radius);

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time elapsed approach 1: %f s\n", time_spent);

  printf("Storing result in FITS\n");
  status = store_data_copying_fits_header(fptr, outfilename, &fits);

  // destroy_fits(&fits);
  fits_close_file(fptr, &status);

  free(data_row);

  if (status) {
    fits_report_error(stderr, status);
  }
  return fits.data;
}

float *approach_3(const char *infilename, const char *outfilename) {
  int status = 0;

  size_t n_iter;
  size_t filter_radius;
  double sigma = 3.5;
  optimal_filter_size_dbl(sigma, &filter_radius, &n_iter);
  printf("sigma: %f filter_radius: %ld niter: %ld\n", sigma, filter_radius,
         n_iter);

  fitsfile *fptr;
  fits_data_t fits;
  if (!fits_open_image(&fptr, infilename, READONLY, &status)) {
    status = extract_data_from_fits(fptr, &fits);
  } else {
    fits_report_error(stderr, status);
    abort();
  }

  float *data = fits.data;
  size_t size_x = fits.naxes[0];
  size_t size_y = fits.naxes[1];

  float *data_row =
      (float *)malloc(sizeof(float) * (size_x + 2 * filter_radius));

  clock_t begin = clock();

  filter_gauss_2d_avx(data, NULL, data_row, NULL, size_x, size_y, n_iter,
                     filter_radius);

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time elapsed approach 1: %f s\n", time_spent);

  printf("Storing result in FITS\n");
  status = store_data_copying_fits_header(fptr, outfilename, &fits);

  // destroy_fits(&fits);
  fits_close_file(fptr, &status);

  free(data_row);

  if (status) {
    fits_report_error(stderr, status);
  }
  return fits.data;
}

void compare_1iter() {
  float exp_data[] = {
      1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0,  3.0,  3.0,  3.0,  4.0, 4.0,
      4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0,  6.0,  7.0,  7.0,  7.0, 7.0,
      8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0,
  };
  float act_data[] = {
      1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0,  3.0,  3.0,  3.0,  4.0, 4.0,
      4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0,  6.0,  7.0,  7.0,  7.0, 7.0,
      8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0,
  };
  size_t size_x = 4;
  size_t size_y = 10;

  size_t n_iter;
  size_t filter_radius;
  double sigma = 3.5;
  optimal_filter_size_dbl(sigma, &filter_radius, &n_iter);
  n_iter = 1;
  printf("sigma: %f filter_radius: %ld niter: %ld\n", sigma, filter_radius,
         n_iter);

  /////////////////// Approach 1
  float *column = (float *)malloc(sizeof(float) * size_y);
  float *data_col =
      (float *)malloc(sizeof(float) * (size_y + 2 * filter_radius));
  float *data_row =
      (float *)malloc(sizeof(float) * (size_x + 2 * filter_radius));

  filter_gauss_col_1(exp_data, column, data_row, data_col, size_x, size_y,
                     n_iter, filter_radius);

  for (size_t i = 0; i < size_y * size_x; i++)
    printf("%.1f ", exp_data[i]);
  printf("\n");

  filter_gauss_col_2(act_data, NULL, NULL, NULL, size_x, size_y, n_iter,
                     filter_radius);

  for (size_t i = 0; i < size_y * size_x; i++)
    printf("%.1f ", act_data[i]);
  printf("\n");

  free(column);
  free(data_col);
  free(data_row);
}
