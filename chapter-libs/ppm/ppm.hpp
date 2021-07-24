#pragma once

// Ported from https://github.com/skeeto/sort-circle

#include <cstdint>
#include <ostream>
#include <vector>

struct RGB {
  uint8_t r, g, b;
  RGB(uint32_t hex)
      : r((hex >> 16) & 0xff), g((hex >> 8) & 0xff), b(hex & 0xff){};
  RGB(uint8_t r, uint8_t g, uint8_t b)
      : r(r), g(g), b(b) {};
};

class PPM {
  std::size_t width, height;
  std::vector<std::uint8_t> buf;
  const float R0;  // dot inner radius
  const float R1;  // dot outer radius
  const float PAD; // message padding
public:
  PPM(size_t width, size_t height);
  void write(std::ostream &stream);
  void set(int x, int y, RGB color);
  RGB get(int x, int y);
  void dot(float x, float y, RGB fgc);
  void set_char(int c, int x, int y, RGB fgc);
  void write_line(std::string message, int x, int y, RGB color);
};
