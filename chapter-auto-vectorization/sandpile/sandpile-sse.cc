#include <cstdio>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <string>
#include <xmmintrin.h>

namespace FractalSSE {
class Sandpile {
private:
  size_t pixel;
  size_t points;
  size_t size;
  char *state;
  char *buffer;
  inline size_t resolveIdx(size_t y, size_t x) { return y * points + x; }

public:
  // N + 2 + 16 to avoid the problem of boundary
  Sandpile(size_t pixel)
      : pixel(pixel), points(pixel + 2), size(points * points),
        state(new char[points * points]), buffer(new char[points * points]){};
  ~Sandpile() {
    if (state != nullptr) {
      delete[] state;
    }
    if (buffer != nullptr) {
      delete[] buffer;
    }
  }

  void stabilize();

  void computeIdentity() {
    std::fprintf(stderr, "WITH_SSE\n");
    // f(ones(n)*6 - f(ones(n)*6)
    for (size_t y = 1; y <= pixel; ++y) {
      for (size_t x = 1; x <= pixel; ++x) {
        buffer[resolveIdx(y, x)] = 6;
      }
    }

    stabilize();

    for (size_t y = 1; y <= pixel; ++y) {
      for (size_t x = 1; x <= pixel; ++x) {
        buffer[resolveIdx(y, x)] = 6 - state[resolveIdx(y, x)];
      }
    }

    stabilize();
  }

  void serialize(FILE *stream) {
    uint8_t imgbuf[3 * pixel * pixel];
    const uint32_t color[] = {0xf53d3d, 0x3d93f5, 0xf53d93, 0x99f53d};
    for (std::int32_t j = 1; j <= pixel; ++j) {
      for (std::int32_t i = 1; i <= pixel; ++i) {
        std::int32_t y = j - 1;
        std::int32_t x = i - 1;
        std::int8_t sand = state[resolveIdx(j, i)];
        uint32_t pxlclr = sand < 4 ? color[sand] : 0xffffff;
        imgbuf[y * 3 * pixel + 3 * x + 0] = pxlclr >> 16;
        imgbuf[y * 3 * pixel + 3 * x + 1] = pxlclr >> 8;
        imgbuf[y * 3 * pixel + 3 * x + 2] = pxlclr >> 0;
      }
    }
    printf("P6\n%zu %zu\n255\n", pixel, pixel);
    fwrite(imgbuf, sizeof(imgbuf), 1, stream);
  }
};

void Sandpile::stabilize() {
  while (1) {

    size_t spills = 0;
    for (size_t y = 1; y <= pixel; ++y) {
      for (size_t x = 1; x <= pixel; x += 16) {
        // Since there is no way to operate over each byte in the vector,
        // we sub then blend based on compare result
        __m128i sand =
            _mm_loadu_si128((const __m128i_u *)(buffer + resolveIdx(y, x)));
        // No gte for epi8
        __m128i cmp = _mm_cmpgt_epi8(sand, _mm_set1_epi8(3));
        __m128i diff = _mm_sub_epi8(sand, _mm_set1_epi8(4));
        __m128i newSand = _mm_blendv_epi8(sand, diff, cmp);

        // Non zero spills
        spills += !_mm_testz_si128(cmp, cmp);

        // Spill over from neighbours
        // Need to increment newSand, follow the same pattern of add then
        // blend based on cmp
        sand =
            _mm_loadu_si128((const __m128i_u *)(buffer + resolveIdx(y - 1, x)));
        cmp = _mm_cmpgt_epi8(sand, _mm_set1_epi8(3));
        __m128i sum = _mm_add_epi8(newSand, _mm_set1_epi8(1));
        newSand = _mm_blendv_epi8(newSand, sum, cmp);

        sand =
            _mm_loadu_si128((const __m128i_u *)(buffer + resolveIdx(y + 1, x)));
        cmp = _mm_cmpgt_epi8(sand, _mm_set1_epi8(3));
        sum = _mm_add_epi8(newSand, _mm_set1_epi8(1));
        newSand = _mm_blendv_epi8(newSand, sum, cmp);

        sand =
            _mm_loadu_si128((const __m128i_u *)(buffer + resolveIdx(y, x - 1)));
        cmp = _mm_cmpgt_epi8(sand, _mm_set1_epi8(3));
        sum = _mm_add_epi8(newSand, _mm_set1_epi8(1));
        newSand = _mm_blendv_epi8(newSand, sum, cmp);

        sand =
            _mm_loadu_si128((const __m128i_u *)(buffer + resolveIdx(y, x + 1)));
        cmp = _mm_cmpgt_epi8(sand, _mm_set1_epi8(3));
        sum = _mm_add_epi8(newSand, _mm_set1_epi8(1));
        newSand = _mm_blendv_epi8(newSand, sum, cmp);

        _mm_storeu_si128((__m128i_u *)(state + resolveIdx(y, x)), newSand);
      }
    }

    std::swap(buffer, state);
    if (!spills) {
      return;
    }
  }
}

} // namespace FractalSSE
