#include "perlin.hpp"

#include <cmath>
#include <cstdint>

namespace Noise {

void Perlin::seedNoise(float seed) {
  if (seed > 0 && seed < 1) {
    // Scale the seed out
    seed *= 65536;
  }

  uint32_t iseed = floor(seed);
  if (iseed < 256) {
    iseed |= iseed << 8;
  }

  for (int i = 0; i < 256; i++) {
    uint8_t v;
    if (i & 1) {
      v = permutation[i] ^ (iseed & 255);
    } else {
      v = permutation[i] ^ ((iseed >> 8) & 255);
    }

    perm[i] = perm[i + 256] = v;
    gradP[i] = gradP[i + 256] = grad3[v % 12];
  }
}

// 2D Perlin Noise
float Perlin::noise2D(float x, float y) {
  uint32_t X = floor(x);
  uint32_t Y = floor(y);

  x = x - X;
  y = y - Y;
  X = X & 255;
  Y = Y & 255;

  float n00 = gradP[X + perm[Y]].dot2(x, y);
  float n01 = gradP[X + perm[Y + 1]].dot2(x, y - 1);
  float n10 = gradP[X + 1 + perm[Y]].dot2(x - 1, y);
  float n11 = gradP[X + 1 + perm[Y + 1]].dot2(x - 1, y - 1);

  float u = fade(x);

  return lerp(lerp(n00, n10, u), lerp(n01, n11, u), fade(y));
};

// 3D Perlin Noise
float Perlin::noise3D(float x, float y, float z) {
  uint32_t X = floor(x);
  uint32_t Y = floor(y);
  uint32_t Z = floor(z);

  x = x - X;
  y = y - Y;
  z = z - Z;
  X = X & 255;
  Y = Y & 255;
  Z = Z & 255;

  float n000 = gradP[X + perm[Y + perm[Z]]].dot3(x, y, z);
  float n001 = gradP[X + perm[Y + perm[Z + 1]]].dot3(x, y, z - 1);
  float n010 = gradP[X + perm[Y + 1 + perm[Z]]].dot3(x, y - 1, z);
  float n011 = gradP[X + perm[Y + 1 + perm[Z + 1]]].dot3(x, y - 1, z - 1);
  float n100 = gradP[X + 1 + perm[Y + perm[Z]]].dot3(x - 1, y, z);
  float n101 = gradP[X + 1 + perm[Y + perm[Z + 1]]].dot3(x - 1, y, z - 1);
  float n110 = gradP[X + 1 + perm[Y + 1 + perm[Z]]].dot3(x - 1, y - 1, z);
  float n111 =
      gradP[X + 1 + perm[Y + 1 + perm[Z + 1]]].dot3(x - 1, y - 1, z - 1);

  float u = fade(x);
  float v = fade(y);
  float w = fade(z);

  return lerp(lerp(lerp(n000, n100, u), lerp(n001, n101, u), w),
              lerp(lerp(n010, n110, u), lerp(n011, n111, u), w), v);
};

} // namespace Noise
