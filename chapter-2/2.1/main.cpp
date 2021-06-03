#include <cstdlib>
#include <iostream>

#include "gol.hpp"
#include "life.hpp"

int main() {

  GliderGun *life = new GliderGun(10, 5);

  GameOfLife *game = new GameOfLife(life);

  for(int i = 0; i < 50; ++i) {
    game->print();
    game->iterate();
    std::cout<<std::endl;
    system("sleep 0.2");
    system("clear");
  }
}
