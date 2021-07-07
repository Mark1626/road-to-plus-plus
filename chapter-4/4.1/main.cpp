#include "perlin.hpp"

#include <cstdint>
#include <cstdio>
#include <array>

int main() {
  const int HEIGHT = 100;
  const int WIDTH = 100;
  Noise::Perlin noise(100);
  std::array<float, (HEIGHT * WIDTH)> noiseMap = {};

  #pragma omp parallel for
  for (int x = 0; x < HEIGHT; ++x) {
    for (int y = 0; y < WIDTH; ++y) {
      int idx = (x * WIDTH) + y;
      noiseMap[idx] = noise.noise2D(x * 0.1f, y * 0.1f);
        // noiseMap[idx] = 1.0f;//noise.noise2D(x * 0.1f, y * 0.1f);
    }
  }

// #ifdef VERBOSE
//   for (uint32_t i = 0; i < HEIGHT * WIDTH; ++i) {
//     std::printf("%f ", noiseMap[i]);
//   }
//   std::printf("\n");
// #endif
}
