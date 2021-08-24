#include <cstdio>
#include <string>

namespace Fractal {

class Sandpile {
private:
  size_t pixels;
  size_t points;
  size_t size;
  char *state;
  char *buffer;
  inline size_t resolveIdx(size_t y, size_t x) { return y * points + x; }
  void stabilize();

public:
  // N + 2 + 16 to avoid the problem of boundary
  Sandpile(size_t pixel)
      : pixels(pixel), points(pixel + 2 + 16), size(points * points),
        state(new char[points * points]), buffer(new char[points * points]){};
  ~Sandpile() {
    if (state != nullptr) {
      delete[] state;
    }
    if (buffer != nullptr) {
      delete[] buffer;
    }
  }
  void computeIdentity() {
    std::fprintf(stderr, "WITH_Normal\n");
    // f(ones(n)*6 - f(ones(n)*6)
    for (size_t y = 1; y <= pixels; ++y) {
      for (size_t x = 1; x <= pixels; ++x) {
        buffer[resolveIdx(y, x)] = 6;
      }
    }

    stabilize();

    for (size_t y = 1; y <= pixels; ++y) {
      for (size_t x = 1; x <= pixels; ++x) {
        buffer[resolveIdx(y, x)] = 6 - state[resolveIdx(y, x)];
      }
    }

    stabilize();
  }

  void serialize(FILE *stream) {
    uint8_t imgbuf[3 * pixels * pixels];
    const uint32_t color[] = {0xf53d3d, 0x3d93f5, 0xf53d93, 0x99f53d};
    for (std::int32_t j = 1; j <= pixels; ++j) {
      for (std::int32_t i = 1; i <= pixels; ++i) {
        std::int32_t y = j - 1;
        std::int32_t x = i - 1;
        std::int8_t sand = state[resolveIdx(j, i)];
        uint32_t pxlclr = sand < 4 ? color[sand] : 0xffffff;
        imgbuf[y * 3 * pixels + 3 * x + 0] = pxlclr >> 16;
        imgbuf[y * 3 * pixels + 3 * x + 1] = pxlclr >> 8;
        imgbuf[y * 3 * pixels + 3 * x + 2] = pxlclr >> 0;
      }
    }
    printf("P6\n%zu %zu\n255\n", pixels, pixels);
    fwrite(imgbuf, sizeof(imgbuf), 1, stream);
  }
};

void Sandpile::stabilize() {
  // fprintf(stderr, "Stabilizing \n");

  while (1) {
    size_t spills = 0;
    for (size_t y = 1; y <= pixels; ++y) {
      for (size_t x = 1; x <= pixels; ++x) {
        char currSand = buffer[resolveIdx(y, x)];
        char newSand = currSand >= 4 ? currSand - 4 : currSand;
        spills += currSand >= 4;
        // Spill over from neighbours
        newSand += buffer[resolveIdx((y - 1), x)] >= 4;
        newSand += buffer[resolveIdx((y + 1), x)] >= 4;
        newSand += buffer[resolveIdx(y, (x - 1))] >= 4;
        newSand += buffer[resolveIdx(y, (x + 1))] >= 4;

        state[resolveIdx(y, x)] = newSand;
      }
    }

    // print();
    std::swap(buffer, state);
    if (!spills) {
      return;
    }
  }
}
} // namespace Fractal
