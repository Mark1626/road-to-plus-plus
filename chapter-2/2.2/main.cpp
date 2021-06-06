#include "gol.hpp"
#include <cstdlib>
#include <ctime>

int main() {
  srand(time(NULL));

  size_t N = 1 << 4;
#ifdef SIZE
  N = 1 << SIZE;
#endif
  Life::GOL game(N);
  game.populate_random();
  // game.print();
  game.compute(1000);
#ifdef DEBUG
  game.print();
#else
  game.serializePPM();
#endif
  return EXIT_SUCCESS;
}
