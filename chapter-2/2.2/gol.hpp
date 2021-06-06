#pragma once

#include <cstdint>
#include <cstring>

namespace Life {
class GOL {
private:
  size_t pixels;
  size_t points;
  size_t size;
  uint8_t *state;
  uint8_t *buffer;
  void iterate();
  void sse_iterate();
  inline size_t resolveIdx(size_t y, size_t x) { return y * points + x; }

public:
  // +2 for boundary, +16 for filling in SSE
  GOL(size_t pixels)
      : pixels(pixels), points(2 + pixels + 16),
        size(points * points),
        state(new uint8_t[size]),
        buffer(new uint8_t[size]) {}
  void compute(size_t generations);
  void populate_random();
  void serializePPM();
  void print();
  ~GOL() {
    if (state != nullptr) {
      delete [] state;
    }
    if (buffer != nullptr) {
      delete [] buffer;
    }
  }
};
} // namespace Life
