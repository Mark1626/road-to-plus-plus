#include "sha1.hpp"
#include <cstdint>
#include <cstring>

// For Endian Conversion
#include <libkern/OSByteOrder.h>

#define SHA1_K0  0x5a827999
#define SHA1_K20 0x6ed9eba1
#define SHA1_K40 0x8f1bbcdc
#define SHA1_K60 0xca62c1d6

namespace Digest {
  // https://en.wikipedia.org/wiki/SHA-1#SHA-1_pseudocode
  void SHA1::hashBlock() {
    uint32_t a,b,c,d,e,t;
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    for(uint8_t i = 0; i < 80; ++i) {
      // Message schedule: extend the sixteen 32-bit words into eighty 32-bit words:
      if (i >= 16) {
        t = buffer[(i+13)&15] ^ buffer[(i+8)&15] ^ buffer[(i+2)&15] ^ buffer[i&15];
        buffer[i&15] = rol32(t, 1);
      }
      if (i < 20) {
        t = (d ^ (b & (c ^ d))) + SHA1_K0;
      } else if (i < 40) {
        t = (b ^ c ^ d) + SHA1_K20;
      } else if (i < 60) {
        t = ((b & c) | (d & (b | c))) + SHA1_K40;
      } else {
        t = (b ^ c ^ d) + SHA1_K60;
      }
      t += rol32(a, 5) + e + buffer[i&15];
      e = d;
      d = c;
      c = rol32(b, 30);
      b = a;
      a = t;
    }
    // Add chunk's hash to result
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
  }

  void SHA1::addUncounted(uint8_t data) {
    uint8_t* const b = (uint8_t*) buffer;
    #ifdef __LITTLE_ENDIAN__
      b[bufferOffset ^ 3] = data;
    #else
      b[bufferOffset] = data;
    #endif
    bufferOffset++;
    if (bufferOffset == 64) {
      hashBlock();
      bufferOffset = 0;
    }
  }

  void SHA1::updateByte(uint8_t data) {
    ++byteCount;
    addUncounted(data);
  }

  void SHA1::update(const char *data, size_t length) {
    for(size_t i = 0; i < length; ++i)
      updateByte(data[i]);
  }

  // Pad bytes to a 64 byte or 512 bit block
  void SHA1::padBlock() {
    // append the bit '1' to the message e.g. by adding 0x80 if message length is a multiple of 8 bits.
    addUncounted(0x80);
    while(bufferOffset != 56)
      addUncounted(0x00);

    addUncounted(0);
    addUncounted(0);
    addUncounted(0);
    addUncounted(byteCount >> 29);
    addUncounted(byteCount >> 21);
    addUncounted(byteCount >> 13);
    addUncounted(byteCount >> 5);
    addUncounted(byteCount << 3);
  }

  uint8_t* SHA1::digest() {
    padBlock();

    #ifdef __LITTLE_ENDIAN__
      state[0] = OSSwapHostToBigInt32(state[0]);
      state[1] = OSSwapHostToBigInt32(state[1]);
      state[2] = OSSwapHostToBigInt32(state[2]);
      state[3] = OSSwapHostToBigInt32(state[3]);
      state[4] = OSSwapHostToBigInt32(state[4]);
    #endif

    return (uint8_t*)state;
  }
}
