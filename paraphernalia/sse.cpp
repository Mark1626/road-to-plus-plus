#include <emmintrin.h>
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <smmintrin.h>
#include <xmmintrin.h>

int main() {

  // Subtract 4 single precision
  __m128 five = _mm_set1_ps(5.0f);
  __m128 one = _mm_set1_ps(1.0f);

  __m128 dst = _mm_sub_ps(five, one);
  float res[4] = {0};
  _mm_storeu_ps(res, dst);
  printf("%f %f\n", res[0], res[1]);

  // Load and add two vectors
  uint8_t test[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  __m128 sandVector = _mm_loadu_si128((const __m128i_u*)test);
  __m128 incVector = _mm_set1_epi8(1);
  __m128 sum = _mm_add_epi8(sandVector, incVector);

  _mm_storeu_si128((__m128i_u*)test, sum);

  for (int i = 0; i < 16; ++i) {
    printf("%d ", test[i]);
  }
  printf("\n");

  // Load with offset and add
  char test2[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  __m128 test2Vector = _mm_loadu_si128((const __m128i_u*)(test2 + 16));
  __m128 sum2 = _mm_add_epi8(test2Vector, incVector);
  _mm_storeu_si128((__m128i_u*)test, sum2);

  for (int i = 0; i < 16; ++i) {
    printf("%d ", test[i]);
  }
  printf("\n");

  // Cmp
  uint8_t arr[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  __m128 v = _mm_loadu_si128((const __m128i_u*)arr);
  __m128 cmp = _mm_cmpgt_epi8(v, _mm_set1_epi8(10));
  // _mm_storeu_si128((__m128i_u*)test, cmp);
  // for (int i = 0; i < 16; ++i) {
  //   printf("%d ", test[i]);
  // }
  // printf("\n");
  int nonzero = !_mm_testz_si128(cmp, cmp);
  printf("%d \n", nonzero);
}
