#include "perlin.hpp"

#include <cstdint>
#include <cstdio>
#include <thread>

#define HEIGHT 10000
#define WIDTH 10000

using namespace Noise;

int main() {
  Perlin *noise = new Perlin(100);
  float* noiseMap = new float[HEIGHT * WIDTH];

  uint32_t x, y;

  #pragma omp parallel for private(y)
  for (x = 0; x < HEIGHT; ++x) {
    for (y = 0; y < WIDTH; ++y) {
      size_t idx = (x * WIDTH) + y;
#ifdef VERBOSE
      std::printf("%d %d %zu\n", x, y, idx);
#endif
      noiseMap[idx] = noise->noise2D(x * 0.1f, y * 0.1f);
    }
  }

#ifdef VERBOSE
  for (uint32_t i = 0; i < HEIGHT * WIDTH; ++i) {
    std::printf("%f ", noiseMap[i]);
  }
  std::printf("\n");
#endif

  delete [] noiseMap;
}
