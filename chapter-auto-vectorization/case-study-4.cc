#include <string>
#include <algorithm>

const int pixel = 1<<8;
const int points = 1<<8;
inline size_t resolveIdx(size_t y, size_t x) { return y * points + x; }

void fn(char* buffer, char* state) {
  int spills = 0;

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
}