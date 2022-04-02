#pragma once

#include <complex>

// This may become templated in the future
namespace simd {

  #define MASK(a, b, c, d) ((a << 6) + (b << 4) + (c << 2) + d)

  // SSE
  template <typename T> struct complex2 { std::complex<T> d[2]; };

  template <> struct complex2<float> { alignas(16) std::complex<float> d[2]; };

  // TODO: Support double
  // AVX
  template <typename T> struct complex4 { std::complex<T> d[4]; };

  template <> struct complex4<float> { alignas(32) std::complex<float> d[4]; };

  // TODO: AVX-512


  //////////////////////// API ///////////////////////////////////////////

  void prod(complex2<float> &c, complex2<float> &a, complex2<float> &b);
  void prod(complex4<float> &c, complex4<float> &a, complex4<float> &b);

  void grid(complex2<float> &c, complex2<float> &a, complex2<float> &b);
  void grid(complex4<float> &c, complex4<float> &a, complex4<float> &b);
}
