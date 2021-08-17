#include <cstdint>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

namespace Fractal {

static const std::int32_t width = 300;
static const std::int32_t height = 200;
static const std::int32_t hpoints = height + 2;
static const std::int32_t wpoints = width + 2;
// static const std::int32_t pixels = 1 << SIZE;
// static const std::int32_t points = pixels + 2;
static const std::int32_t size = hpoints * wpoints;

class Sandpile {
private:
  std::vector<std::int8_t> _state;
  std::vector<std::int8_t> _buffer;
  void stabilize(std::int8_t *__restrict__ state,
                 std::int8_t *__restrict__ buffer);
  inline std::int32_t resolveIdx(std::int32_t y, std::int32_t x) {
    return y * wpoints + x;
  }

public:
  // N + 2 + 16 to avoid the problem of boundary
  Sandpile() : _state(size, 0), _buffer(size, 0){};
  void computeIdentity();
  void serialize(std::ostream &stream) {
    uint8_t imgbuf[3 * width * height];
    const uint32_t color[] = {0xf53d3d, 0x3d93f5, 0xf53d93, 0x99f53d};
    for (std::int32_t j = 1; j <= height; ++j) {
      for (std::int32_t i = 1; i <= width; ++i) {
        std::int32_t y = j - 1;
        std::int32_t x = i - 1;
        std::int8_t sand = _state[resolveIdx(j, i)];
        uint32_t pxlclr = sand < 4 ? color[sand] : 0xffffff;
        imgbuf[y * 3 * width + 3 * x + 0] = pxlclr >> 16;
        imgbuf[y * 3 * width + 3 * x + 1] = pxlclr >> 8;
        imgbuf[y * 3 * width + 3 * x + 2] = pxlclr >> 0;
      }
    }
    stream << "P6\n";
    stream << width << " " << height << "\n";
    stream << "255\n";
    std::copy(std::begin(imgbuf), std::end(imgbuf),
              std::ostream_iterator<uint8_t>(stream));
  }
};

void Sandpile::stabilize(std::int8_t *__restrict__ buffer,
                         std::int8_t *__restrict__ state) {
  while (1) {
    size_t spills = 0;

    #pragma omp parallel for collapse(2)
    for (size_t y = 1; y <= height; ++y) {
      // size_t spills_t = 0;
      for (size_t x = 1; x <= width; ++x) {
        std::int8_t currSand = buffer[resolveIdx(y, x)];
        std::int8_t newSand = currSand >= 4 ? currSand - 4 : currSand;

        #pragma omp critical
        {
          spills += currSand >= 4;
        }

        // Spill over from neighbours
        newSand = newSand + (buffer[resolveIdx((y - 1), x)] >= 4);
        newSand = newSand + (buffer[resolveIdx((y + 1), x)] >= 4);
        newSand = newSand + (buffer[resolveIdx(y, (x - 1))] >= 4);
        newSand = newSand + (buffer[resolveIdx(y, (x + 1))] >= 4);

        state[resolveIdx(y, x)] = newSand;
        // spills = spills + spills_t;
      }
    }

#ifdef ANIMATE
    serialize(std::cout);
#endif

    std::swap(buffer, state);
    if (!spills) {
      return;
    }
  }
}

void Sandpile::computeIdentity() {
  std::cerr << "WITH_Auto\n";
  // f(ones(n)*6 - f(ones(n)*6)
  {
    std::int8_t *buffer = _buffer.data();
    std::int8_t *state = _state.data();

    for (size_t y = 1; y <= height; ++y) {
      for (size_t x = 1; x <= width; ++x) {
        buffer[resolveIdx(y, x)] = 6;
      }
    }

    stabilize(buffer, state);

    for (size_t y = 1; y <= height; ++y) {
      for (size_t x = 1; x <= width; ++x) {
        buffer[resolveIdx(y, x)] = 6 - state[resolveIdx(y, x)];
      }
    }

    stabilize(buffer, state);
  }
}

} // namespace Fractal

int main() {
  Fractal::Sandpile sandpile;
  sandpile.computeIdentity();
  sandpile.serialize(std::cout);

#ifdef ANIMATE
  // 3s delay to close
  for (int i = 0; i < 180; ++i)
    sandpile.serialize(std::cout);
#endif
}
