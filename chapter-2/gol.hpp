#pragma once

#include <cstdint>
#include "life.hpp"

#define WIDTH 50
#define HEIGHT 20

class World {
public:
  uint8_t cell[HEIGHT][WIDTH];
  World() {
    for (int i = 0; i < HEIGHT; ++i) {
      for (int j = 0; j < WIDTH; ++j) {
        cell[i][j] = 0;
      }
    }
  };
};

class GameOfLife {
private:
  World current;
  World buffer;
  void update();
  int nextState(int xcoord, int ycoord);

public:
  GameOfLife();
  GameOfLife(Life *en);
  void iterate();
  void print();
};
