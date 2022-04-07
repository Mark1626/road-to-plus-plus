#include "simd-complex.hh"

#include <immintrin.h>
#include <xmmintrin.h>

namespace simd {

void debug(char *msg, __m128 x) {
  float a[4];
  printf("%s\n", msg);
  _mm_store_ps(a, x);
  for (int i = 0; i < 4; i++) {
    printf("%.1f ", a[i]);
  }
  printf("\n");
}

void prod(complex2<float> &c, complex2<float> &a, complex2<float> &b) {
  // ra1 ia1 ra2 ia2
  float *a_raw = reinterpret_cast<float(&)[4]>(a);
  // rb1 ib1 rb2 ib2
  float *b_raw = reinterpret_cast<float(&)[4]>(b);
  float *c_raw = reinterpret_cast<float(&)[4]>(c);

  __m128 a_vec = _mm_load_ps(a_raw);
  __m128 b_vec = _mm_load_ps(b_raw);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2
  __m128 interm1 = _mm_mul_ps(a_vec, b_vec);

  const float sign[4] = {-1.0, 1.0, -1.0, 1.0};
  const int sample1 = 0b0101;
  const int sample2 = 0b1010;
  const int swap_mask = MASK(2, 3, 0, 1);

  __m128 twist1 = _mm_permute_ps(b_vec, swap_mask);
  // ra1ia2 ra2ia1 rb1ib2 rb2ib1
  __m128 interm2 = _mm_mul_ps(a_vec, twist1);

  // ra1ia1 ia1ib1 rb1ib1 ia2ib2
  __m128 interm3 = _mm_blend_ps(interm1, interm2, sample1);
  // ra1rb1 ra2ia2 ra2ia2 rb2ib2
  __m128 interm4 = _mm_blend_ps(interm1, interm2, sample2);

  // ia1ib1 ra1ia1 ia2ib2 rb1ib1
  interm3 = _mm_permute_ps(interm3, swap_mask);

  __m128 sign_vec = _mm_load_ps(sign);
  // -ia1ib1 ra1ia1 -ia2ib2 rb1ib1
  interm3 = _mm_mul_ps(interm3, sign_vec);

  __m128 res_vec = _mm_add_ps(interm3, interm4);

  _mm_store_ps(c_raw, res_vec);
}

void grid(complex2<float> &c, complex2<float> &a, complex2<float> &b) {
  // ra1 ia1 ra2 ia2
  float *a_raw = reinterpret_cast<float(&)[4]>(a);
  // rb1 ib1 rb2 ib2
  float *b_raw = reinterpret_cast<float(&)[4]>(b);
  float *c_raw = reinterpret_cast<float(&)[4]>(c);

  __m128 a_vec = _mm_loadu_ps(a_raw);
  __m128 b_vec = _mm_loadu_ps(b_raw);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2
  __m128 interm1 = _mm_mul_ps(a_vec, b_vec);

  const float sign[4] = {-1.0, 1.0, -1.0, 1.0};
  const int sample1 = 0b0101;
  const int sample2 = 0b1010;
  const int swap_mask = MASK(2, 3, 0, 1);

  __m128 twist1 = _mm_permute_ps(b_vec, swap_mask);
  // ra1ib1 ia1rb1 ra2ib2 ia2rb2
  __m128 interm2 = _mm_mul_ps(a_vec, twist1);

  // interm 1 : ra1rb1 ia1ib1 ra2ia2 ia2ib2
  // interm 2 : ra1ib1 ia1rb1 ra2ib2 ia2rb2

  // ra1rb1 ia1rb1 ra2ia2 ia2rb2
  __m128 interm3 = _mm_blend_ps(interm1, interm2, sample1);
  // ra1ib1 ia1ib1 ra2ib2 ia2ib2
  __m128 interm4 = _mm_blend_ps(interm1, interm2, sample2);

  // ia1ib1 ra1ib1 ia2ib2 ra2ib2
  interm3 = _mm_permute_ps(interm3, swap_mask);

  __m128 sign_vec = _mm_loadu_ps(sign);
  // -ia1ib1 ra1ib1 -ia2ib2 ra2ib2
  interm3 = _mm_mul_ps(interm3, sign_vec);

  // This final product of complex a * complex b
  __m128 res_vec = _mm_add_ps(interm3, interm4);

  __m128 c_vec = _mm_loadu_ps(c_raw);

  // c = c + a * b
  c_vec = _mm_add_ps(c_vec, res_vec);

  _mm_storeu_ps(c_raw, c_vec);
}
} // namespace simd
