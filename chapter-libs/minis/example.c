#include "ppm.h"
#include <stdio.h>

int main() {
  const int height = 320;
  const int width = 240;
  ppm img = ppm_init(height, width);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      ppm_set(&img, y, x, (RGB){rand() % 255, rand() % 255, rand() % 255});
    }
  }

  ppm_serialize(&img, stdout);

  ppm_destroy(&img);
}
