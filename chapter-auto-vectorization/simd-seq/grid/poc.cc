#include <complex>
#include <immintrin.h>
#include <xmmintrin.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/Matrix.h>

typedef std::complex<float> CFloat;

struct CFloat2 {
  alignas(16) std::complex<float> v[2];
};

template <typename T> class Matrix {
public:
  int M, N;
  T *val;
  Matrix<T>(int M, int N) : M(M), N(N) { val = new T[M * N]; }
  ~Matrix<T>() {
    if (val)
      delete[] val;
  }
  T &operator()(int x, int y) { return val[x * N + y]; }
  T *ptr(int x, int y) { return val[x * N + y]; }
};

namespace simd {

#define MASK(a, b, c, d) ((a << 6) + (b << 4) + (c << 2) + d)

void dotprod(CFloat2 &c, CFloat2 &a, CFloat2 &b) {
  // ra1 ia1 ra2 ia2
  float *a_raw = reinterpret_cast<float(&)[4]>(a);
  // rb1 ib1 rb2 ib2
  float *b_raw = reinterpret_cast<float(&)[4]>(b);
  float *c_raw = reinterpret_cast<float(&)[4]>(c);

  float res_raw[4];

  __m128 a_vec = _mm_loadu_ps(a_raw);
  __m128 b_vec = _mm_loadu_ps(b_raw);

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

  __m128 sign_vec = _mm_loadu_ps(sign);
  // -ia1ib1 ra1ia1 -ia2ib2 rb1ib1
  interm3 = _mm_mul_ps(interm3, sign_vec);

  __m128 res_vec = _mm_add_ps(interm3, interm4);

  _mm_storeu_ps(c_raw, res_vec);

  // c = reinterpret_cast<CFloat2<float>(&)>(res_raw);
}
} // namespace simd

void test_noncasa_access() {
  int N = 16;
  Matrix<CFloat> grid(N, N);
  float a = 3.5f;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      auto real = (double)std::rand() / (double)(RAND_MAX / a);
      auto imag = (double)std::rand() / (double)(RAND_MAX / a);
      grid(i, j) = CFloat(real, imag);
    }
  }

  printf("Access scalar: (%f %f) (%f %f)\n", grid(1, 0).real(),
         grid(1, 0).imag(), grid(1, 1).real(), grid(1, 1).imag());

         printf("Access scalar: (%f %f) (%f %f)\n", grid(1, 2).real(),
         grid(1, 2).imag(), grid(1, 3).real(), grid(1, 3).imag());

  CFloat2 *vec = reinterpret_cast<CFloat2 *>(&grid(1, 0));

  int i = 0;
  printf("Access vector: (%f %f) (%f %f)\n", vec[i].v[0].real(), vec[i].v[0].imag(),
         vec[i].v[1].real(), vec[i].v[1].imag());

  i = 1;
  printf("Access vector: (%f %f) (%f %f)\n", vec[i].v[0].real(), vec[i].v[0].imag(),
         vec[i].v[1].real(), vec[i].v[1].imag());
}

void test_casa_access() {
  int N = 16;
  casacore::Matrix<CFloat> grid(N, N);
  float a = 3.5f;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      auto real = (double)std::rand() / (double)(RAND_MAX / a);
      auto imag = (double)std::rand() / (double)(RAND_MAX / a);
      grid(j, i) = CFloat(real, imag);
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("(%f %f) ", grid(j, i).real(), grid(j, i).imag());
    }
    printf("\n");
  }

  printf("Access scalar: (%f %f) (%f %f)\n", grid(0, 1).real(),
         grid(0, 1).imag(), grid(1, 1).real(), grid(1, 1).imag());

         printf("Access scalar: (%f %f) (%f %f)\n", grid(2, 1).real(),
         grid(2, 1).imag(), grid(3, 1).real(), grid(3, 1).imag());
  
  CFloat2 *vec = reinterpret_cast<CFloat2 *>(&grid(1, 0));

  int i = 0;
  printf("Access vector: (%f %f) (%f %f)\n", vec[i].v[0].real(), vec[i].v[0].imag(),
         vec[i].v[1].real(), vec[i].v[1].imag());

  i = 1;
  printf("Access vector: (%f %f) (%f %f)\n", vec[i].v[0].real(), vec[i].v[0].imag(),
         vec[i].v[1].real(), vec[i].v[1].imag());
}

int main() {
  printf("Access without CASA\n");
  test_noncasa_access();
  printf("Access with CASA\n");
  test_casa_access();
}
