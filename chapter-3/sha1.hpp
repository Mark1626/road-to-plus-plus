#pragma once

#include <cstdint>
#include <cstring>
#include <sys/_types/_size_t.h>

namespace Digest {
class SHA1 {
private:
  uint32_t buffer[16];
  uint32_t state[5];
  uint32_t byteCount;
  int bufferOffset;

  void padBlock();
  void hashBlock();
  void addUncounted(uint8_t data);
  void updateByte(uint8_t data);

  static uint32_t rol32(uint32_t number, uint32_t bits) {
    return (number << bits) | (number >> (32 - bits));
  }

public:
  SHA1()
      : state{0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0},
        byteCount(0), bufferOffset(0){}
  SHA1(const char *data, size_t length) : SHA1() {
    update(data, length);
  }
  void update(const char *data, size_t length);
  uint8_t *digest();
};
} // namespace Digest
