#include <iostream>
#include <fstream>
#include "ppm.hpp"

int main(int argc, char** argv) {
  RGB colors[] = {0x743e66, 0xff7ed2, 0x6dc1ca, 0x87d80a, 0x7d1a85, 0x08a1a3, 0x86b826, 0x849f95, 0x64c3a7, 0xbef27e};

  size_t width = 800;
  size_t height = 600;
  PPM image(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int randi = (int)(rand() % 10);
      image.set(x, y, colors[randi]);
    }
  }

  if (argc > 1) {
    std::ofstream os(argv[1], std::ofstream::out);
    image.write(os);
  } else {
    image.write(std::cout);
  }
}
