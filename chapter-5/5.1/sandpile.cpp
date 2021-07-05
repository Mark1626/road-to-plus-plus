#include "sandpile.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>

#ifdef WITH_SSE
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#endif

namespace Fractal {

void Sandpile::stabilize() {
#ifdef DEBUG
  uint32_t iter = 0;
#endif
fprintf(stderr, "Stabilizing \n");

  while (1) {

#ifdef DEBUG
    ++iter;
    if (iter > 10)
      return;
#endif

    size_t spills = 0;
    for (size_t y = 1; y <= pixel; ++y) {
      for (size_t x = 1; x <= pixel; ++x) {
        char currSand = buffer[resolveIdx(y, x)];
        char newSand = currSand >= 4 ? currSand - 4 : currSand;
        spills += currSand >= 4;
        // Spill over from neighbours
        newSand += buffer[resolveIdx((y - 1), x)] >= 4;
        newSand += buffer[resolveIdx((y + 1), x)] >= 4;
        newSand += buffer[resolveIdx(y, (x - 1))] >= 4;
        newSand += buffer[resolveIdx(y, (x + 1))] >= 4;

        state[resolveIdx(y, x)] = newSand;
      }
    }

    // print();
    std::swap(buffer, state);
    if (!spills) {
      return;
    }
  }
}

#ifdef WITH_SSE
void printV(char* msg, __m128 v) {
  uint8_t test[16] = {0};
  printf("%s \n", msg);
  _mm_storeu_si128((__m128i_u*)test, v);
  for (int i = 0; i < 16; ++i) {
    printf("%d ", test[i]);
  }
  printf("\n\n");
}
#endif

#ifdef WITH_SSE
void Sandpile::sse_stabilize() {
#ifdef DEBUG
  uint32_t iter = 0;
#endif
fprintf(stderr, "Stabilizing with SSE \n");

  while (1) {

#ifdef DEBUG
    ++iter;
    if (iter > 2)
      return;
#endif


    size_t spills = 0;
    for (size_t y = 1; y <= pixel; ++y) {
      for (size_t x = 1; x <= pixel; x += 16) {
        // Since there is no way to operate over each byte in the vector,
        // we sub then blend based on compare result
        __m128 sand =
            _mm_loadu_si128((const __m128i_u *)(buffer + resolveIdx(y, x)));
        // No gte for epi8
        __m128 cmp = _mm_cmpgt_epi8(sand, _mm_set1_epi8(3));
        __m128 diff = _mm_sub_epi8(sand, _mm_set1_epi8(4));
        __m128 newSand = _mm_blendv_epi8(sand, diff, cmp);

        // Non zero spills
        spills += !_mm_testz_si128(cmp, cmp);

        // Spill over from neighbours
        // Need to increment newSand, follow the same pattern of add then blend
        // based on cmp
        sand = _mm_loadu_si128((const __m128i_u *)(buffer + resolveIdx(y - 1, x)));
        cmp = _mm_cmpgt_epi8(sand, _mm_set1_epi8(3));
        __m128 sum = _mm_add_epi8(newSand, _mm_set1_epi8(1));
        newSand = _mm_blendv_epi8(newSand, sum, cmp);

        sand = _mm_loadu_si128((const __m128i_u *)(buffer + resolveIdx(y + 1, x)));
        cmp = _mm_cmpgt_epi8(sand, _mm_set1_epi8(3));
        sum = _mm_add_epi8(newSand, _mm_set1_epi8(1));
        newSand = _mm_blendv_epi8(newSand, sum, cmp);

        sand = _mm_loadu_si128((const __m128i_u *)(buffer + resolveIdx(y, x - 1)));
        cmp = _mm_cmpgt_epi8(sand, _mm_set1_epi8(3));
        sum = _mm_add_epi8(newSand, _mm_set1_epi8(1));
        newSand = _mm_blendv_epi8(newSand, sum, cmp);

        sand = _mm_loadu_si128((const __m128i_u *)(buffer + resolveIdx(y, x + 1)));
        cmp = _mm_cmpgt_epi8(sand, _mm_set1_epi8(3));
        sum = _mm_add_epi8(newSand, _mm_set1_epi8(1));
        newSand = _mm_blendv_epi8(newSand, sum, cmp);

        _mm_storeu_si128((__m128i_u*)(state + resolveIdx(y, x)), newSand);
      }
    }

#ifdef DEBUG
    print();
#endif

#ifdef ANIMATE
  serialize();
#endif

    std::swap(buffer, state);
    if (!spills) {
      return;
    }
  }
}
#endif

void Sandpile::computeIdentity() {
  // f(ones(n)*6 - f(ones(n)*6)
  for (size_t y = 1; y <= pixel; ++y) {
    for (size_t x = 1; x <= pixel; ++x) {
      buffer[resolveIdx(y, x)] = 6;
    }
  }

#ifdef WITH_SSE
  sse_stabilize();
#else
  stabilize();
#endif

  for (size_t y = 1; y <= pixel; ++y) {
    for (size_t x = 1; x <= pixel; ++x) {
      buffer[resolveIdx(y, x)] = 6 - state[resolveIdx(y, x)];
    }
  }

#ifdef WITH_SSE
  sse_stabilize();
#else
  stabilize();
#endif
}

void Sandpile::serialize() {
  uint8_t imgbuf[3 * pixel * pixel];
  const uint32_t color[] = {0xf53d3d, 0x3d93f5, 0xf53d93, 0x99f53d};
  for (size_t j = 1; j <= pixel; ++j) {
    for (size_t i = 1; i <= pixel; ++i) {
      size_t y = j - 1;
      size_t x = i - 1;
      uint8_t sand = state[resolveIdx(j, i)];
      uint32_t pxlclr = sand < 4 ? color[sand] : 0xffffff;
      imgbuf[y * 3 * pixel + 3 * x + 0] = pxlclr >> 16;
      imgbuf[y * 3 * pixel + 3 * x + 1] = pxlclr >> 8;
      imgbuf[y * 3 * pixel + 3 * x + 2] = pxlclr >> 0;
    }
  }
  printf("P6\n%zu %zu\n255\n", pixel, pixel);
  fwrite(imgbuf, sizeof(imgbuf), 1, stdout);
}

void Sandpile::print() {
  for (size_t y = 0; y < points; ++y) {
    for (size_t x = 0; x < points; ++x) {
      uint8_t sand = state[resolveIdx(y, x)];
      printf("%hhu ", sand);
    }
#ifdef DEBUG
    printf("\t");
    for (size_t x = 0; x < points; ++x) {
      uint8_t sand = buffer[resolveIdx(y, x)];
      printf("%hhu ", sand);
    }
#endif
    printf("\n");
  }
  printf("\n");
}

} // namespace Fractal
