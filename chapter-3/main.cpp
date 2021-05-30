#include "sha1.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>

int main() {
  Digest::SHA1 sha1;
  char str[] = "Hello World";
  sha1.update(str, strlen(str));
  const uint8_t *msgDigest = sha1.digest();
  for (int i = 0; i < 20; ++i) {
    printf("%02x", (uint32_t)msgDigest[i]);
  }
  std::cout << std::endl;
}
