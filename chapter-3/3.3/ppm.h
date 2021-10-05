#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef PPM_H
#define PPM_H

typedef struct {
  unsigned char r;
  unsigned char g;
  unsigned char b;
} RGB;

typedef struct {
  int width;
  int height;
  unsigned char *buff;
#ifdef PPM_FLOATS
  float R1;
  float R0;
#endif
} ppm;

static ppm ppm_init(const int width, const int height) {
  return (ppm){.width = width,
               .height = height,
               .buff = (unsigned char *)malloc(3 * width * height *
                                               sizeof(unsigned char)),
#ifdef PPM_FLOATS
               .R1 = (width / 400.0f),
               .R0 = (width / 200.0f)
#endif
  };
}

static void ppm_destroy(ppm *img) { free(img->buff); }

static void ppm_serialize(ppm *img, FILE *stream) {
  fprintf(stream, "P6\n%d %d\n255\n", img->width, img->height);
  fwrite(img->buff, 3 * img->width * img->height * sizeof(unsigned char), 1,
         stream);
}

static void ppm_set(ppm *img, int x, int y, RGB color) {
  img->buff[y * img->width * 3 + x * 3 + 0] = color.r;
  img->buff[y * img->width * 3 + x * 3 + 1] = color.g;
  img->buff[y * img->width * 3 + x * 3 + 2] = color.b;
}

static RGB ppm_get(ppm *img, int x, int y) {
  return (RGB){img->buff[y * img->width * 3 + x * 3 + 0],
               img->buff[y * img->width * 3 + x * 3 + 1],
               img->buff[y * img->width * 3 + x * 3 + 2]};
}

#ifdef PPM_FLOATS

static float clamp(float x, float lower, float upper) {
  if (x < lower)
    return lower;
  if (x > upper)
    return upper;
  return x;
}

static float smoothstep(float lower, float upper, float x) {
  x = clamp((x - lower) / (upper - lower), 0.0f, 1.0f);
  return x * x * (3.0f - 2.0f * x);
}

static void ppm_dot(ppm *img, float x, float y, RGB fgc) {
  float fr = fgc.r, fg = fgc.g, fb = fgc.b;

  int miny = floorf(y - img->R1 - 1);
  int maxy = ceilf(y + img->R1 + 1);
  int minx = floorf(x - img->R1 - 1);
  int maxx = ceilf(x + img->R1 + 1);

  for (int py = miny; py <= maxy; py++) {
    float dy = py - y;
    for (int px = minx; px <= maxx; px++) {
      float dx = px - x;
      float d = sqrtf(dy * dy + dx * dx);
      float a = smoothstep(img->R1, img->R0, d);

      RGB bgc = ppm_get(img, px, py);
      float br = bgc.r, bg = bgc.g, bb = bgc.b;

      float r = a * fr + (1 - a) * br;
      float g = a * fg + (1 - a) * bg;
      float b = a * fb + (1 - a) * bb;
      ppm_set(img, px, py,
              {(unsigned char)r, (unsigned char)g, (unsigned char)b});
    }
  }
}

#endif

// v in range -1.0 to 1.0
static unsigned long hue(double v) {
    unsigned long h = (int)((v + 1)*179.5) / 60;
    unsigned long f = (int)((v + 1)*179.5) % 60;
    unsigned long t = 0xff * f / 60;
    unsigned long q = 0xff - t;
    switch (h) {
    case 0: return 0xff0000UL | (t << 8);
    case 1: return (q << 16) | 0x00ff00UL;
    case 2: return 0x00ff00UL | t;
    case 3: return (q << 8) | 0x0000ffUL;
    case 4: return (t << 16) | 0x0000ffUL;
    case 5: return 0xff0000UL | q;
    }
    abort();
}

#endif
