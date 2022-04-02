#include <complex>
#include <immintrin.h>
#include <xmmintrin.h>

void debug(char *msg, __m128 x) {
  float a[4];
  printf("%s\n", msg);
  _mm_store_ps(a, x);
  for (int i = 0; i < 4; i++) {
    printf("%.1f ", a[i]);
  }
  printf("\n");
}

void debug(char *msg, __m256 x) {
  float a[8];
  printf("%s\n", msg);
  _mm256_store_ps(a, x);
  for (int i = 0; i < 8; i++) {
    printf("%.1f ", a[i]);
  }
  printf("\n");
}


namespace simd {
template <typename T> struct complex2 { std::complex<T> d[2]; };

template <> struct complex2<float> { alignas(16) std::complex<float> d[2]; };

#define MASK(a, b, c, d) ((a << 6) + (b << 4) + (c << 2) + d)

void prod(complex2<float>& c, complex2<float> &a, complex2<float> &b) {
  // ra1 ia1 ra2 ia2
  float *a_raw = reinterpret_cast<float(&)[4]>(a);
  // rb1 ib1 rb2 ib2
  float *b_raw = reinterpret_cast<float(&)[4]>(b);
  float *c_raw = reinterpret_cast<float(&)[4]>(c);
  float res_raw[4];

  __m128 a_vec = _mm_load_ps(a_raw);
  __m128 b_vec = _mm_load_ps(b_raw);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2
  __m128 interm1 = _mm_mul_ps(a_vec, b_vec);

  const float sign[4] = {-1.0, 1.0, -1.0, 1.0};
  const int sample1 = 0b0101;
  const int sample2 = 0b1010;
  const int swap_mask = MASK(2, 3, 0, 1);

  __m128 twist1 = _mm_blend_ps(a_vec, b_vec, sample1);
  __m128 twist2 = _mm_blend_ps(a_vec, b_vec, sample2);

  twist1 = _mm_permute_ps(twist1, swap_mask);

  // ra1ia1 ra2ia2 rb1ib1 rb2ib2
  __m128 interm2 = _mm_mul_ps(twist1, twist2);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2
  // ra1ia1 ra2ia2 rb1ib1 rb2ib2

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

  // c = reinterpret_cast<complex2<float>(&)>(res_raw);
}
} // namespace simd

namespace simd {
template <typename T> struct complex4 { std::complex<T> d[4]; };

template <> struct complex4<float> { alignas(32) std::complex<float> d[4]; };

#define MASK_8(a, b, c, d, e, f, g, h) ((a<<14) + (b << 12) + (c << 10) + (d << 8) + (e << 6) + (f << 4) + (g << 2) + h)
#define MASK_4(a, b, c, d) ((a << 6) + (b << 4) + (c << 2) + d)

void prod(complex4<float> &c, complex4<float> &a, complex4<float> &b) {
  // ra1 ia1 ra2 ia2 ra3 ia3 ra4 ia4
  float *a_raw = reinterpret_cast<float(&)[8]>(a);
  // rb1 ib1 rb2 ib2 rb3 ib3 rb4 ib4
  float *b_raw = reinterpret_cast<float(&)[8]>(b);
  float res_raw[8];

  __m256 a_vec = _mm256_load_ps(a_raw);
  __m256 b_vec = _mm256_load_ps(b_raw);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2 ra3rb3 ia3ib3 ra4ia4 ia4ib4
  __m256 interm1 = _mm256_mul_ps(a_vec, b_vec);

  const float sign[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
  const int sample1 = 0b01010101;
  const int sample2 = 0b10101010;
  const int swap_mask = MASK_4(2, 3, 0, 1);

  __m256 twist1 = _mm256_blend_ps(a_vec, b_vec, sample1);
  __m256 twist2 = _mm256_blend_ps(a_vec, b_vec, sample2);

  twist1 = _mm256_permute_ps(twist1, swap_mask);

  // ra1ia1 ra2ia2 rb1ib1 rb2ib2
  __m256 interm2 = _mm256_mul_ps(twist1, twist2);

  // ra1rb1 ia1ib1 ra2ia2 ia2ib2
  // ra1ia1 ra2ia2 rb1ib1 rb2ib2

  // ra1ia1 ia1ib1 rb1ib1 ia2ib2
  __m256 interm3 = _mm256_blend_ps(interm1, interm2, sample1);
  // ra1rb1 ra2ia2 ra2ia2 rb2ib2
  __m256 interm4 = _mm256_blend_ps(interm1, interm2, sample2);

  // ia1ib1 ra1ia1 ia2ib2 rb1ib1
  interm3 = _mm256_permute_ps(interm3, swap_mask);

  __m256 sign_vec = _mm256_load_ps(sign);
  // -ia1ib1 ra1ia1 -ia2ib2 rb1ib1
  interm3 = _mm256_mul_ps(interm3, sign_vec);

  __m256 res_vec = _mm256_add_ps(interm3, interm4);

  _mm256_store_ps(res_raw, res_vec);

  c = reinterpret_cast<complex4<float>(&)>(res_raw);
}
} // namespace simd

int main() {
  int N = 128;
  std::complex<float> *a = new std::complex<float>[N];
  std::complex<float> *b = new std::complex<float>[N];
  std::complex<float> *c = new std::complex<float>[N];

  float lim = 5.0f;
  for (int i = 0; i < N; i++) {
    auto real = (double)std::rand() / (double)(RAND_MAX / lim);
    auto imag = (double)std::rand() / (double)(RAND_MAX / lim);
    a[i] = std::complex<float>(real, imag);

    real = (double)std::rand() / (double)(RAND_MAX / lim);
    imag = (double)std::rand() / (double)(RAND_MAX / lim);
    b[i] = std::complex<float>(real, imag);
  }

  {
    simd::complex2<float> *a_vec = reinterpret_cast<simd::complex2<float>*>(a);
    simd::complex2<float> *b_vec = reinterpret_cast<simd::complex2<float>*>(b);
    simd::complex2<float> *c_vec = reinterpret_cast<simd::complex2<float>*>(c);

    for (int i = 0; i < N / 2; i++) {
      simd::prod(c_vec[i], a_vec[i], b_vec[i]);
    }

    printf("%f %f\n", c[0].real(), c[0].imag());
  }

  {
    simd::complex4<float> *a_vec = reinterpret_cast<simd::complex4<float>*>(a);
    simd::complex4<float> *b_vec = reinterpret_cast<simd::complex4<float>*>(b);
    simd::complex4<float> *c_vec = reinterpret_cast<simd::complex4<float>*>(c);

    for (int i = 0; i < N / 4; i++) {
      simd::prod(c_vec[i], a_vec[i], b_vec[i]);
    }

    printf("%f %f\n", c[0].real(), c[0].imag());
  }

  delete [] a;
  delete [] b;
  delete [] c;
}
