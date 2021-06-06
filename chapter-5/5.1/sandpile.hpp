#pragma once

#include <cstdint>
#include <cstring>

namespace Fractal {
class Sandpile {
private:
  size_t pixel;
  size_t points;
  size_t size;
  char *state;
  char *buffer;
  void stabilize();
  void sse_stabilize();
  inline size_t resolveIdx(size_t y, size_t x) { return y * points + x; }

public:
  // N + 2 + 16 to avoid the problem of boundary
  Sandpile(size_t pixel)
      : pixel(pixel), points(pixel + 2 + 16), size(points * points),
        state(new char[points * points]),
        buffer(new char[points * points]){};
  void computeIdentity();
  void serialize();
  void print();
  ~Sandpile() {
    if (state != nullptr) {
      delete[] state;
    }
    if (buffer != nullptr) {
      delete[] buffer;
    }
  }
};
} // namespace Fractal
