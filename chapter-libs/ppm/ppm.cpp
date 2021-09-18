
// Ported from https://github.com/skeeto/sort-circle
#include <cstdint>
#include <iterator>
#include <ostream>
#include <vector>

#include "font.hpp"
#include "ppm.hpp"

float clamp(float x, float lower, float upper) {
  if (x < lower)
    return lower;
  if (x > upper)
    return upper;
  return x;
}

float smoothstep(float lower, float upper, float x) {
  x = clamp((x - lower) / (upper - lower), 0.0f, 1.0f);
  return x * x * (3.0f - 2.0f * x);
}

PPM::PPM(size_t width, size_t height)
    : width(width), height(height), buf(3 * width * height, 0),
      R0((width / 400.0f)), R1((width / 200.0f)), PAD(width / 128.0f){};

void PPM::write(std::ostream &stream) {
  stream << "P6\n";
  stream << width << " " << height << "\n";
  stream << "255\n";
  std::copy(std::begin(buf), std::end(buf),
            std::ostream_iterator<uint8_t>(stream));
}

void PPM::set(int x, int y, RGB color) {
  buf[y * width * 3 + x * 3 + 0] = color.r;
  buf[y * width * 3 + x * 3 + 1] = color.g;
  buf[y * width * 3 + x * 3 + 2] = color.b;
}

RGB PPM::get(int x, int y) {
  unsigned long r = buf[y * width * 3 + x * 3 + 0];
  unsigned long g = buf[y * width * 3 + x * 3 + 1];
  unsigned long b = buf[y * width * 3 + x * 3 + 2];
  return RGB(r, g, b);
}

void PPM::dot(float x, float y, RGB fgc) {
  float fr = fgc.r, fg = fgc.g, fb = fgc.b;

  int miny = floorf(y - R1 - 1);
  int maxy = ceilf(y + R1 + 1);
  int minx = floorf(x - R1 - 1);
  int maxx = ceilf(x + R1 + 1);

  for (int py = miny; py <= maxy; py++) {
    float dy = py - y;
    for (int px = minx; px <= maxx; px++) {
      float dx = px - x;
      float d = sqrtf(dy * dy + dx * dx);
      float a = smoothstep(R1, R0, d);

      RGB bgc = get(px, py);
      float br = bgc.r, bg = bgc.g, bb = bgc.b;

      float r = a * fr + (1 - a) * br;
      float g = a * fg + (1 - a) * bg;
      float b = a * fb + (1 - a) * bb;
      set(px, py, RGB(r, g, b));
    }
  }
}

void PPM::set_char(int c, int x, int y, RGB fgc) {
  float fr = fgc.r, fg = fgc.g, fb = fgc.b;
  for (int dy = 0; dy < FONT_H; dy++) {
    for (int dx = 0; dx < FONT_W; dx++) {
      float a = font_value(c, dx, dy);
      if (a > 0.0f) {
        RGB bgc = get(x + dx, y + dy);
        float br = bgc.r, bg = bgc.g, bb = bgc.b;

        float r = a * fr + (a - 1) * br;
        float g = a * fg + (a - 1) * bg;
        float b = a * fb + (a - 1) * bb;
        set(x + dx, y + dy, RGB(r, g, b));
      }
    }
  }
}

void PPM::write_line(std::string message, int x, int y, RGB color) {
  for (int c = 0; message[c]; c++)
    set_char(message[c], x + (c * FONT_W + PAD), y + PAD, color);
}
