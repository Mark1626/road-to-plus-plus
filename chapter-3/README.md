# Chapter 3: SHA-1

[Back](../README.md)

Implementing SHA-1 and see if I can optimize something with a low level call. Reference for the SHA-1 taken from an old code I had as well as the Wikipedia psuedo code, only targetting MacOS

Assertion is the main file itself
## Learning

Endian Conversion with `OSByteOrder.h`, this probably gets converted into `bswap` or `movbe`, need to check which instruction

> Update: In Mac it's bswap

```cpp
   state[i] = (((state[i]) << 24) & 0xff000000) |
      (((state[i]) << 8) & 0x00ff0000) |
      (((state[i]) >> 8) & 0x0000ff00) |
      (((state[i]) >> 24) & 0x000000ff)

  // Replaced with MacOS OSByteOrder.h
  state[i] = OSSwapHostToBigInt32(state[i]);
```

## Reference

- [Byte Swapping](https://en.wikipedia.org/wiki/Endianness#Byte_swapping)
