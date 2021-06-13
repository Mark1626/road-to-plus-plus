#include "sandpile.hpp"
#include <cstdint>

int main() {
  size_t pixels =  1 << 4;

#ifdef SIZE
  pixels = 1 << SIZE;
#endif

  Fractal::Sandpile sandpile(pixels);
  sandpile.computeIdentity();
#ifdef DEBUG
  sandpile.print();
#else
  sandpile.serialize();
#endif

#ifdef ANIMATE
// 3s delay to close
for(int i = 0; i < 180; ++i)
  sandpile.serialize();
#endif
}
