#include "simd-complex.hh"
#include <assert.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/BasicSL/Complexfwd.h>
#include <chrono>
#include <complex>
#include <immintrin.h>
#include <iomanip>
#include <string>
#include <xmmintrin.h>

// #define DEBUG
bool assert = false;
bool avx_gridding = true;
bool sse_gridding = true;
bool ptr_gridding = true;
bool std_gridding = true;

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

void assert_complex(std::string msg, CFloat expected, CFloat actual) {
  float exp = expected.real();
  float act = actual.real();
  float tolerance = 0.1f;

  if (fabs(exp - act) > tolerance) {
    std::cout << msg << " expected " << expected << " but was " << actual
              << std::endl;
    // std::cout << "Expected real part to be " << exp << " but was " << act
    //           << std::endl;
    assert(fabs(exp - act) < tolerance);
  }

  exp = expected.imag();
  act = actual.imag();
  if (fabs(exp - act) > tolerance) {
    std::cout << msg << " expected " << expected << " but was " << actual
              << std::endl;
    // std::cerr << "Expected imaginary part to be " << exp << " but was " <<
    // act
    //           << std::endl;
    assert(fabs(exp - act) < tolerance);
  }
}

// CASA has a wierd access to array
void gridding_casa_std(casacore::Matrix<casacore::Complex> &grid,
                       casacore::Matrix<casacore::Complex> &convFunc,
                       const casacore::Complex &cVis, const int iu,
                       const int iv, const int support) {
  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    for (int suppu = -support; suppu <= support; suppu++) {
      const int uoff = suppu + support;
      casacore::Complex wt = convFunc(uoff, voff);
      grid(iu + suppu, iv + suppv) += cVis * wt;
    }
  }
}

void gridding_casa_ptr(casacore::Matrix<casacore::Complex> &grid,
                       casacore::Matrix<casacore::Complex> &conv,
                       const casacore::Complex &cVis, const int iu,
                       const int iv, const int support) {

  float rVis = cVis.real();
  float iVis = cVis.imag();
  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    const int uoff = 0;
    float *wtPtrF = reinterpret_cast<float *>(&conv(uoff, voff));
    float *gridPtrF =
        reinterpret_cast<float *>(&grid(iu - support, iv + suppv));
    for (int suppu = -support; suppu <= support;
         suppu++, wtPtrF += 2, gridPtrF += 2) {
      gridPtrF[0] += rVis * wtPtrF[0] - iVis * wtPtrF[1];
      gridPtrF[1] += rVis * wtPtrF[1] + iVis * wtPtrF[0];
    }
  }
}

void gridding_std(Matrix<CFloat> &grid, Matrix<CFloat> &convFunc,
                  const CFloat &cVis, const int iu, const int iv,
                  const int support) {
#ifdef DEBUG
  std::cerr << "Gridding std" << std::endl;
#endif
  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    for (int suppu = -support; suppu <= support; suppu++) {
      const int uoff = suppu + support;
      CFloat wt = convFunc(voff, uoff);
#ifdef DEBUG
      std::cerr << cVis << "*" << wt << "+" << grid(iv + suppv, iu + suppu)
                << " " << iv + suppv << " , " << iu + suppu << std::endl;
#endif
      grid(iv + suppv, iu + suppu) += cVis * wt;
    }
  }
}

void gridding_simd_2(Matrix<CFloat> &grid, Matrix<CFloat> &convFunc,
                     const CFloat &cVis, const int iu, const int iv,
                     const int support) {
#ifdef DEBUG
  std::cout << "Gridding simd" << std::endl;
#endif
  simd::complex2<float> cvis_vec = {.d = {cVis, cVis}};

  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    int uoff = 0;

    simd::complex2<float> *conv_vec =
        reinterpret_cast<simd::complex2<float> *>(&convFunc(voff, uoff));
    simd::complex2<float> *grid_vec = reinterpret_cast<simd::complex2<float> *>(
        &grid(iv + suppv, iu - support));
    ;

    // 2n+1, the last one can be done separately
    for (int i = 0; i < support; i++) {
#ifdef DEBUG
      std::cout << cvis_vec.d[0] << "*" << conv_vec[i].d[0] << "+"
                << grid_vec[i].d[0] << std::endl;
      std::cout << cvis_vec.d[1] << "*" << conv_vec[i].d[1] << "+"
                << grid_vec[i].d[1] << std::endl;
#endif

      simd::grid(grid_vec[i], conv_vec[i], cvis_vec);
    }

    // Last grid point
    int suppu = support;
    uoff = suppu + support;
    CFloat wt = convFunc(voff, uoff);
    grid(iv + suppv, iu + suppu) += cVis * wt;
  }
}

void gridding_simd_4(Matrix<CFloat> &grid, Matrix<CFloat> &convFunc,
                     const CFloat &cVis, const int iu, const int iv,
                     const int support) {
#ifdef DEBUG
  std::cout << "Gridding simd avx" << std::endl;
#endif
  simd::complex4<float> cvis_vec = {.d = {cVis, cVis, cVis, cVis}};

  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    int uoff = 0;

    simd::complex4<float> *conv_vec =
        reinterpret_cast<simd::complex4<float> *>(&convFunc(voff, uoff));
    simd::complex4<float> *grid_vec = reinterpret_cast<simd::complex4<float> *>(
        &grid(iv + suppv, iu - support));
    ;

    int rem = (2 * support + 1) % 4;
    int tiles = (2 * support + 1) / 4;

    // 2n+1, the last one can be done separately
    for (int i = 0; i < tiles; i++) {
#ifdef DEBUG
      std::cout << cvis_vec.d[0] << "*" << conv_vec[i].d[0] << "+"
                << grid_vec[i].d[0] << " " << iv + suppv << " , "
                << iu - support << std::endl;
      std::cout << cvis_vec.d[1] << "*" << conv_vec[i].d[1] << "+"
                << grid_vec[i].d[1] << " " << iv + suppv << " , "
                << iu - support << std::endl;
      std::cout << cvis_vec.d[2] << "*" << conv_vec[i].d[2] << "+"
                << grid_vec[i].d[2] << " " << iv + suppv << " , "
                << iu - support << std::endl;
      std::cout << cvis_vec.d[3] << "*" << conv_vec[i].d[3] << "+"
                << grid_vec[i].d[3] << " " << iv + suppv << " , "
                << iu - support << std::endl;
#endif
      simd::grid(grid_vec[i], conv_vec[i], cvis_vec);
    }

    for (int i = 1; i <= rem; i++) {
      // Last grid point
      int suppu = support - rem + i;
      uoff = suppu + support;
      CFloat wt = convFunc(voff, uoff);
#ifdef DEBUG
      std::cout << cVis << "*" << wt << "+" << grid(iv + suppv, iu + suppu)
                << std::endl;
#endif
      grid(iv + suppv, iu + suppu) += cVis * wt;
    }
  }
}

void gridding_casa_simd_2(casacore::Matrix<casacore::Complex> &grid,
                          casacore::Matrix<casacore::Complex> &convFunc,
                          const casacore::Complex &cVis, const int iu,
                          const int iv, const int support) {
#ifdef DEBUG
  std::cout << "Gridding simd 2 casa" << std::endl;
#endif
  simd::complex2<float> cvis_vec = {.d = {cVis, cVis}};

  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    int uoff = 0;

    simd::complex2<float> *conv_vec =
        reinterpret_cast<simd::complex2<float> *>(&convFunc(uoff, voff));
    simd::complex2<float> *grid_vec = reinterpret_cast<simd::complex2<float> *>(
        &grid(iu - support, iv + suppv));
    ;

    // 2n+1, the last one can be done separately
    for (int i = 0; i < support; i++) {
#ifdef DEBUG
      std::cout << cvis_vec.d[0] << "*" << conv_vec[i].d[0] << "+"
                << grid_vec[i].d[0] << std::endl;
      std::cout << cvis_vec.d[1] << "*" << conv_vec[i].d[1] << "+"
                << grid_vec[i].d[1] << std::endl;
#endif

      simd::grid(grid_vec[i], conv_vec[i], cvis_vec);
    }

    // Last grid point
    int suppu = support;
    uoff = suppu + support;
    casacore::Complex wt = convFunc(uoff, voff);
    grid(iu + suppu, iv + suppv) += cVis * wt;
  }
}

void gridding_casa_simd_4(casacore::Matrix<casacore::Complex> &grid,
                          casacore::Matrix<casacore::Complex> &convFunc,
                          const casacore::Complex &cVis, const int iu,
                          const int iv, const int support) {
#ifdef DEBUG
  std::cout << "Gridding simd avx casa" << std::endl;
#endif
  simd::complex4<float> cvis_vec = {.d = {cVis, cVis, cVis, cVis}};

  for (int suppv = -support; suppv <= support; suppv++) {
    const int voff = suppv + support;
    int uoff = 0;

    simd::complex4<float> *conv_vec =
        reinterpret_cast<simd::complex4<float> *>(&convFunc(uoff, voff));
    simd::complex4<float> *grid_vec = reinterpret_cast<simd::complex4<float> *>(
        &grid(iu - support, iv + suppv));
    ;

    int rem = (2 * support + 1) % 4;
    int tiles = (2 * support + 1) / 4;

    // 2n+1, the last one can be done separately
    for (int i = 0; i < tiles; i++) {
#ifdef DEBUG
      std::cout << cvis_vec.d[0] << "*" << conv_vec[i].d[0] << "+"
                << grid_vec[i].d[0] << " " << iv + suppv << " , "
                << iu - support << std::endl;
      std::cout << cvis_vec.d[1] << "*" << conv_vec[i].d[1] << "+"
                << grid_vec[i].d[1] << " " << iv + suppv << " , "
                << iu - support << std::endl;
      std::cout << cvis_vec.d[2] << "*" << conv_vec[i].d[2] << "+"
                << grid_vec[i].d[2] << " " << iv + suppv << " , "
                << iu - support << std::endl;
      std::cout << cvis_vec.d[3] << "*" << conv_vec[i].d[3] << "+"
                << grid_vec[i].d[3] << " " << iv + suppv << " , "
                << iu - support << std::endl;
#endif
      simd::grid(grid_vec[i], conv_vec[i], cvis_vec);
    }

    for (int i = 1; i <= rem; i++) {
      // Last grid point
      int suppu = support - rem + i;
      uoff = suppu + support;
      CFloat wt = convFunc(uoff, voff);
#ifdef DEBUG
      std::cout << cVis << "*" << wt << "+" << grid(iu + suppu, iv + suppv)
                << std::endl;
#endif
      grid(iu + suppu, iv + suppv) += cVis * wt;
    }
  }
}

void test_simple_sse() {
  std::cout << "Test SSE" << std::endl;

  CFloat a[2] = {CFloat(1.0f, 2.0f), CFloat(3.0f, 4.0f)};
  CFloat b[2] = {CFloat(5.0f, 6.0f), CFloat(7.0f, 8.0f)};

  CFloat c_exp[2] = {CFloat(0.0f, 0.0f), CFloat(0.0f, 0.0f)};

  c_exp[0] = a[0] * b[0];
  c_exp[1] = a[1] * b[1];

  CFloat c_act[2] = {CFloat(0.0f, 0.0f), CFloat(0.0f, 0.0f)};

  simd::complex2<float> *a_vec = reinterpret_cast<simd::complex2<float> *>(a);
  simd::complex2<float> *b_vec = reinterpret_cast<simd::complex2<float> *>(b);
  simd::complex2<float> *c_vec =
      reinterpret_cast<simd::complex2<float> *>(c_act);

  simd::prod(c_vec[0], a_vec[0], b_vec[0]);

  assert_complex("Assert first val", c_exp[0], c_act[0]);
  assert_complex("Assert second val", c_exp[1], c_act[1]);

  ///////////////////////////////////////////////////

  int N = 128;
  CFloat a_arr[N];
  CFloat b_arr[N];

  float range = 3.2f;
  for (int i = 0; i < N; i++) {
    auto real = (double)std::rand() / (double)(RAND_MAX / range);
    auto imag = (double)std::rand() / (double)(RAND_MAX / range);
    a_arr[i] = CFloat(real, imag);

    real = (double)std::rand() / (double)(RAND_MAX / range);
    imag = (double)std::rand() / (double)(RAND_MAX / range);
    b_arr[i] = CFloat(real, imag);
  }

  CFloat c_arr_exp[N];

#ifdef DEBUG
  std::cout << "Scalar" << std::endl;
#endif
  for (int i = 0; i < N; i++) {
    c_arr_exp[i] = a_arr[i] * b_arr[i];
#ifdef DEBUG
    std::cout << a_arr[i] << "*" << b_arr[i] << "=" << c_arr_exp[i]
              << std::endl;
#endif
  }

  CFloat c_arr_actual[N];

  simd::complex2<float> *a_arr_vec =
      reinterpret_cast<simd::complex2<float> *>(a_arr);
  simd::complex2<float> *b_arr_vec =
      reinterpret_cast<simd::complex2<float> *>(b_arr);
  simd::complex2<float> *c_arr_vec =
      reinterpret_cast<simd::complex2<float> *>(c_arr_actual);

#ifdef DEBUG
  std::cout << "Vector" << std::endl;
#endif
  for (int i = 0; i < N / 2; i++) {
    prod(c_arr_vec[i], a_arr_vec[i], b_arr_vec[i]);
#ifdef DEBUG
    std::cout << a_arr_vec[i].d[0] << "*" << b_arr_vec[i].d[0] << "="
              << c_arr_vec[i].d[0] << std::endl;
    std::cout << a_arr_vec[i].d[0] << "*" << b_arr_vec[i].d[1] << "="
              << c_arr_vec[i].d[1] << std::endl;
#endif
  }

  for (int i = 0; i < N; i++) {
    // asc_arr_exp[i] = a_arr[i] * b_arr[i];
    assert_complex("Idx " + std::to_string(i), c_arr_exp[i], c_arr_actual[i]);
  }
  std::cout << "Assertions passed" << std::endl;
}

void test_simple_avx() {
  std::cout << "Test AVX" << std::endl;

  CFloat a[4] = {CFloat(1.0f, 2.0f), CFloat(3.0f, 4.0f), CFloat(1.5f, 2.5f),
                 CFloat(3.5f, 4.5f)};
  CFloat b[4] = {CFloat(5.0f, 6.0f), CFloat(7.0f, 8.0f), CFloat(5.5f, 6.5f),
                 CFloat(7.5f, 8.5f)};

  CFloat c_exp[4] = {CFloat(0.0f, 0.0f), CFloat(0.0f, 0.0f), CFloat(0.0f, 0.0f),
                     CFloat(0.0f, 0.0f)};

  c_exp[0] = a[0] * b[0];
  c_exp[1] = a[1] * b[1];
  c_exp[2] = a[2] * b[2];
  c_exp[3] = a[3] * b[3];

  CFloat c_act[4] = {CFloat(0.0f, 0.0f), CFloat(0.0f, 0.0f), CFloat(0.0f, 0.0f),
                     CFloat(0.0f, 0.0f)};

  simd::complex4<float> *a_vec = reinterpret_cast<simd::complex4<float> *>(a);
  simd::complex4<float> *b_vec = reinterpret_cast<simd::complex4<float> *>(b);
  simd::complex4<float> *c_vec =
      reinterpret_cast<simd::complex4<float> *>(c_act);

  simd::prod(c_vec[0], a_vec[0], b_vec[0]);

  assert_complex("Assert first val", c_exp[0], c_act[0]);
  assert_complex("Assert second val", c_exp[1], c_act[1]);
  assert_complex("Assert first val", c_exp[2], c_act[2]);
  assert_complex("Assert second val", c_exp[3], c_act[3]);

  ///////////////////////////////////////////////////

  int N = 128;
  CFloat a_arr[N];
  CFloat b_arr[N];

  float range = 3.2f;
  for (int i = 0; i < N; i++) {
    auto real = (double)std::rand() / (double)(RAND_MAX / range);
    auto imag = (double)std::rand() / (double)(RAND_MAX / range);
    a_arr[i] = CFloat(real, imag);

    real = (double)std::rand() / (double)(RAND_MAX / range);
    imag = (double)std::rand() / (double)(RAND_MAX / range);
    b_arr[i] = CFloat(real, imag);
  }

  CFloat c_arr_exp[N];

#ifdef DEBUG
  std::cout << "Scalar" << std::endl;
#endif
  for (int i = 0; i < N; i++) {
    c_arr_exp[i] = a_arr[i] * b_arr[i];
#ifdef DEBUG
    std::cerr << a_arr[i] << "*" << b_arr[i] << "=" << c_arr_exp[i]
              << std::endl;
#endif
  }

  CFloat c_arr_actual[N];

  simd::complex4<float> *a_arr_vec =
      reinterpret_cast<simd::complex4<float> *>(a_arr);
  simd::complex4<float> *b_arr_vec =
      reinterpret_cast<simd::complex4<float> *>(b_arr);
  simd::complex4<float> *c_arr_vec =
      reinterpret_cast<simd::complex4<float> *>(c_arr_actual);

#ifdef DEBUG
  std::cout << "Vector" << std::endl;
#endif
  for (int i = 0; i < N / 4; i++) {
    prod(c_arr_vec[i], a_arr_vec[i], b_arr_vec[i]);
#ifdef DEBUG
    std::cout << a_arr_vec[i].d[0] << "*" << b_arr_vec[i].d[0] << "="
              << c_arr_vec[i].d[0] << std::endl;
    std::cout << a_arr_vec[i].d[1] << "*" << b_arr_vec[i].d[1] << "="
              << c_arr_vec[i].d[1] << std::endl;
    std::cout << a_arr_vec[i].d[2] << "*" << b_arr_vec[i].d[2] << "="
              << c_arr_vec[i].d[2] << std::endl;
    std::cout << a_arr_vec[i].d[3] << "*" << b_arr_vec[i].d[3] << "="
              << c_arr_vec[i].d[3] << std::endl;
#endif
  }

  for (int i = 0; i < N; i++) {
    // asc_arr_exp[i] = a_arr[i] * b_arr[i];
    assert_complex("Idx " + std::to_string(i), c_arr_exp[i], c_arr_actual[i]);
  }
  std::cout << "Assertions passed" << std::endl;
}

void test_noncasa_access(int N, int convN) {

  float a = 5.0;
  Matrix<CFloat> conv(convN, convN);
  for (int i = 0; i < convN; i++) {
    for (int j = 0; j < convN; j++) {
      auto real = (double)std::rand() / (double)(RAND_MAX / a);
      auto imag = (double)std::rand() / (double)(RAND_MAX / a);
      conv(i, j) = CFloat(real, imag);
    }
  }

  int support = (convN / 2) - 1;

  Matrix<CFloat> visibility(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      auto real = (double)std::rand() / (double)(RAND_MAX / a);
      auto imag = (double)std::rand() / (double)(RAND_MAX / a);
      visibility(i, j) = CFloat(real, imag);
    }
  }

#ifdef DEBUG_ACCESS
  printf("Access scalar: (%f %f) (%f %f)\n", grid_std(1, 0).real(),
         grid_std(1, 0).imag(), grid_std(1, 1).real(), grid_std(1, 1).imag());

  printf("Access scalar: (%f %f) (%f %f)\n", grid_std(1, 2).real(),
         grid_std(1, 2).imag(), grid_std(1, 3).real(), grid_std(1, 3).imag());

  CFloat2 *vec = reinterpret_cast<CFloat2 *>(&grid_std(1, 0));

  int i = 0;
  printf("Access vector: (%f %f) (%f %f)\n", vec[i].v[0].real(),
         vec[i].v[0].imag(), vec[i].v[1].real(), vec[i].v[1].imag());

  i = 1;
  printf("Access vector: (%f %f) (%f %f)\n", vec[i].v[0].real(),
         vec[i].v[0].imag(), vec[i].v[1].real(), vec[i].v[1].imag());
#endif

  int offset = support;

  Matrix<CFloat> grid_std(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      grid_std(i, j) = CFloat(0.0f, 0.0f);
    }
  }

#ifdef DEBUG
  for (int i = 0; i < convN; i++) {
    for (int j = 0; j < convN; j++) {
      std::cout << conv(i, j) << " ";
    }
    std::cout << std::endl;
  }
#endif

  {
    // Gridding all points
    for (int u = 0; u < N - 2 * offset; u++) {
      for (int v = 0; v < N - 2 * offset; v++) {
        int iu = u + offset;
        int iv = v + offset;
        CFloat cvis = visibility(iu, iv);
        gridding_std(grid_std, conv, cvis, u + offset, v + offset, support);
#ifdef DEBUG
        std::cerr << "Grid" << std::endl;
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < N; j++) {
            std::cerr << std::setprecision(2) << grid_std(i, j) << " ";
          }
          std::cerr << std::endl;
        }
#endif
      }
    }
  }

  Matrix<CFloat> grid_1(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      grid_1(i, j) = CFloat(0.0f, 0.0f);
    }
  }

  {
    // Gridding all points
    for (int u = 0; u < N - 2 * offset; u++) {
      for (int v = 0; v < N - 2 * offset; v++) {
        int iu = u + offset;
        int iv = v + offset;
        CFloat cvis = visibility(iu, iv);
        gridding_simd_2(grid_1, conv, cvis, u + offset, v + offset, support);
      }
    }
  }

  Matrix<CFloat> grid_2(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      grid_2(i, j) = CFloat(0.0f, 0.0f);
    }
  }

  {
    // Gridding all points
    for (int u = 0; u < N - 2 * offset; u++) {
      for (int v = 0; v < N - 2 * offset; v++) {
        int iu = u + offset;
        int iv = v + offset;
        CFloat cvis = visibility(iu, iv);
        gridding_simd_4(grid_2, conv, cvis, u + offset, v + offset, support);

#ifdef DEBUG
        std::cout << "Grid" << std::endl;
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < N; j++) {
            std::cout << std::setprecision(2) << grid_2(i, j) << " ";
          }
          std::cout << std::endl;
        }
#endif
      }
    }
  }

  if (assert) {
    CFloat exp_c, act_c;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        // ASKAPCHECK(1==2, "1!=2");
        exp_c = grid_std(j, i);
        act_c = grid_1(j, i);

        assert_complex("Asserting point: " + std::to_string(i) + " " +
                           std::to_string(j),
                       exp_c, act_c);
        // ASKAPLOG_INFO_STR(logger, "A: " << grid_std(j, i) << " ");
        // ASKAPLOG_INFO_STR(logger, "E: " << grid_expected(j, i) << " ");
      }
    }
    std::cout << "All assertions passed for SSE" << std::endl;
  }

  if (assert) {
    CFloat exp_c, act_c;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        // ASKAPCHECK(1==2, "1!=2");
        exp_c = grid_std(j, i);
        act_c = grid_2(j, i);

        assert_complex("Asserting point: " + std::to_string(i) + " " +
                           std::to_string(j),
                       exp_c, act_c);
        // ASKAPLOG_INFO_STR(logger, "A: " << grid_std(j, i) << " ");
        // ASKAPLOG_INFO_STR(logger, "E: " << grid_expected(j, i) << " ");
      }
    }
    std::cout << "All assertions passed for AVX" << std::endl;
  }
}

void test_casa_access(int N, int convN) {

  float a = 5.0;
  casacore::Matrix<casacore::Complex> conv(convN, convN);
  for (int i = 0; i < convN; i++) {
    for (int j = 0; j < convN; j++) {
      auto real = (double)std::rand() / (double)(RAND_MAX / a);
      auto imag = (double)std::rand() / (double)(RAND_MAX / a);
      conv(i, j) = casacore::Complex(real, imag);
    }
  }

  int support = (convN / 2) - 1;

  casacore::Matrix<casacore::Complex> visibility(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      auto real = (double)std::rand() / (double)(RAND_MAX / a);
      auto imag = (double)std::rand() / (double)(RAND_MAX / a);
      visibility(i, j) = casacore::Complex(real, imag);
    }
  }

#ifdef DEBUG_ACCESS
  printf("Access scalar: (%f %f) (%f %f)\n", grid_std(1, 0).real(),
         grid_std(1, 0).imag(), grid_std(1, 1).real(), grid_std(1, 1).imag());

  printf("Access scalar: (%f %f) (%f %f)\n", grid_std(1, 2).real(),
         grid_std(1, 2).imag(), grid_std(1, 3).real(), grid_std(1, 3).imag());

  casacore::Complex *vec =
      reinterpret_cast<casacore::Complex *>(&grid_std(1, 0));

  int i = 0;
  printf("Access vector: (%f %f) (%f %f)\n", vec[i].v[0].real(),
         vec[i].v[0].imag(), vec[i].v[1].real(), vec[i].v[1].imag());

  i = 1;
  printf("Access vector: (%f %f) (%f %f)\n", vec[i].v[0].real(),
         vec[i].v[0].imag(), vec[i].v[1].real(), vec[i].v[1].imag());
#endif

  int offset = support;

  casacore::Matrix<casacore::Complex> grid_std(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      grid_std(i, j) = casacore::Complex(0.0f, 0.0f);
    }
  }

#ifdef DEBUG
  for (int i = 0; i < convN; i++) {
    for (int j = 0; j < convN; j++) {
      std::cout << conv(i, j) << " ";
    }
    std::cout << std::endl;
  }
#endif

  if (std_gridding) {
    auto start = std::chrono::steady_clock::now();

    // Gridding all points
    for (int u = 0; u < N - 2 * offset; u++) {
      for (int v = 0; v < N - 2 * offset; v++) {
        int iu = u + offset;
        int iv = v + offset;
        casacore::Complex cvis = visibility(iu, iv);
        gridding_casa_std(grid_std, conv, cvis, u + offset, v + offset,
                          support);
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "CPP Wallclock Std gridding "
              << (std::chrono::duration<double, std::milli>(diff).count())
              << " ms" << std::endl;
  }

  casacore::Matrix<casacore::Complex> grid_ptr(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      grid_ptr(i, j) = casacore::Complex(0.0f, 0.0f);
    }
  }

  if (ptr_gridding) {
    auto start = std::chrono::steady_clock::now();

    // Gridding all points
    for (int u = 0; u < N - 2 * offset; u++) {
      for (int v = 0; v < N - 2 * offset; v++) {
        int iu = u + offset;
        int iv = v + offset;
        casacore::Complex cvis = visibility(iu, iv);
        gridding_casa_ptr(grid_ptr, conv, cvis, u + offset, v + offset,
                          support);
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "CPP Wallclock ptr gridding "
              << (std::chrono::duration<double, std::milli>(diff).count())
              << " ms" << std::endl;
  }

  casacore::Matrix<casacore::Complex> grid_1(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      grid_1(i, j) = casacore::Complex(0.0f, 0.0f);
    }
  }

  if (sse_gridding) {
    auto start = std::chrono::steady_clock::now();

    // Gridding all points
    for (int u = 0; u < N - 2 * offset; u++) {
      for (int v = 0; v < N - 2 * offset; v++) {
        int iu = u + offset;
        int iv = v + offset;
        casacore::Complex cvis = visibility(iu, iv);
        gridding_casa_simd_2(grid_1, conv, cvis, u + offset, v + offset,
                             support);
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "CPP Wallclock SIMD 2 Tiled gridding "
              << (std::chrono::duration<double, std::milli>(diff).count())
              << " ms" << std::endl;
  }

  casacore::Matrix<casacore::Complex> grid_2(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      grid_2(i, j) = casacore::Complex(0.0f, 0.0f);
    }
  }

  if (avx_gridding) {
    auto start = std::chrono::steady_clock::now();

    // Gridding all points
    for (int u = 0; u < N - 2 * offset; u++) {
      for (int v = 0; v < N - 2 * offset; v++) {
        int iu = u + offset;
        int iv = v + offset;
        casacore::Complex cvis = visibility(iu, iv);
        gridding_casa_simd_4(grid_2, conv, cvis, u + offset, v + offset,
                             support);
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "CPP Wallclock SIMD 4 Tiled gridding "
              << (std::chrono::duration<double, std::milli>(diff).count())
              << " ms" << std::endl;
  }

  if (assert) {
    casacore::Complex exp_c, act_c;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        // ASKAPCHECK(1==2, "1!=2");
        exp_c = grid_std(j, i);
        act_c = grid_1(j, i);

        assert_complex("Asserting point: " + std::to_string(i) + " " +
                           std::to_string(j),
                       exp_c, act_c);
        // ASKAPLOG_INFO_STR(logger, "A: " << grid_std(j, i) << " ");
        // ASKAPLOG_INFO_STR(logger, "E: " << grid_expected(j, i) << " ");
      }
    }
    std::cout << "All assertions passed for SSE" << std::endl;
  }

  if (assert) {
    casacore::Complex exp_c, act_c;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        // ASKAPCHECK(1==2, "1!=2");
        exp_c = grid_std(j, i);
        act_c = grid_2(j, i);

        assert_complex("Asserting point: " + std::to_string(i) + " " +
                           std::to_string(j),
                       exp_c, act_c);
        // ASKAPLOG_INFO_STR(logger, "A: " << grid_std(j, i) << " ");
        // ASKAPLOG_INFO_STR(logger, "E: " << grid_expected(j, i) << " ");
      }
    }
    std::cout << "All assertions passed for AVX" << std::endl;
  }
}

int main() {

  int N = 1024;
  int convN = 256;

  // test_simple_sse();
  // test_simple_avx();

  // casacore::Matrix<casacore::Complex> grid(N, N);

  // printf("Access without CASA\n");
  // test_noncasa_access(N, convN);

  printf("Access with CASA\n");
  test_casa_access(N, convN);
  // test_casa_access();
}
