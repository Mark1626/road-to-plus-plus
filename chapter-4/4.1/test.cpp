#include "perlin.hpp"
#include <cstdio>

using namespace Noise;

int main() {
  Perlin *noise = new Perlin(50);
  printf("%f \n", noise->noise2D(1.0f, 2.1f));
}