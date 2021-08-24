#include <cstdint>
#include <string>
#include <vector>

#include "Config.h"

namespace FractalAutoVec {

static const std::int32_t pixels = 1 << SIZE;
static const std::int32_t points = pixels + 2;
static const std::int32_t size = points * points;

class Sandpile {
private:
  std::vector<std::int8_t> _state;
  std::vector<std::int8_t> _buffer;
  void stabilize(std::int8_t *__restrict__ state,
                 std::int8_t *__restrict__ buffer);
  inline std::int32_t resolveIdx(std::int32_t y, std::int32_t x) {
    return y * points + x;
  }

public:
  // N + 2 + 16 to avoid the problem of boundary
  Sandpile() : _state(size, 0), _buffer(size, 0){};
  void computeIdentity();
  void serialize(FILE *stream) {
    uint8_t imgbuf[3 * pixels * pixels];
    const uint32_t color[] = {0xf53d3d, 0x3d93f5, 0xf53d93, 0x99f53d};
    for (std::int32_t j = 1; j <= pixels; ++j) {
      for (std::int32_t i = 1; i <= pixels; ++i) {
        std::int32_t y = j - 1;
        std::int32_t x = i - 1;
        std::int8_t sand = _state[resolveIdx(j, i)];
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

void Sandpile::stabilize(std::int8_t *__restrict__ buffer,
                         std::int8_t *__restrict__ state) {
  while (1) {
    size_t spills = 0;
    for (size_t y = 1; y <= pixels; ++y) {
      for (size_t x = 1; x <= pixels; ++x) {
        std::int8_t currSand = buffer[resolveIdx(y, x)];
        std::int8_t newSand = currSand >= 4 ? currSand - 4 : currSand;
        spills += currSand >= 4;
        // Spill over from neighbours
        newSand = newSand + (buffer[resolveIdx((y - 1), x)] >= 4);
        newSand = newSand + (buffer[resolveIdx((y + 1), x)] >= 4);
        newSand = newSand + (buffer[resolveIdx(y, (x - 1))] >= 4);
        newSand = newSand + (buffer[resolveIdx(y, (x + 1))] >= 4);

        state[resolveIdx(y, x)] = newSand;
      }
    }

    std::swap(buffer, state);
    if (!spills) {
      return;
    }
  }
}

void Sandpile::computeIdentity() {
  std::fprintf(stderr, "WITH_Auto\n");
  // f(ones(n)*6 - f(ones(n)*6)

  std::int8_t *buffer = _buffer.data();
  std::int8_t *state = _state.data();

  for (size_t y = 1; y <= pixels; ++y) {
    for (size_t x = 1; x <= pixels; ++x) {
      buffer[resolveIdx(y, x)] = 6;
    }
  }

  stabilize(buffer, state);

  for (size_t y = 1; y <= pixels; ++y) {
    for (size_t x = 1; x <= pixels; ++x) {
      buffer[resolveIdx(y, x)] = 6 - state[resolveIdx(y, x)];
    }
  }

  stabilize(buffer, state);
}

} // namespace FractalAutoVec
