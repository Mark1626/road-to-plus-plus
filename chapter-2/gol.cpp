#include "gol.hpp"
#include <cstdlib>
#include <iostream>

#include "life.hpp"

GameOfLife::GameOfLife() {
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      current.cell[i][j] = rand() % 2;
    }
  }
}

GameOfLife::GameOfLife(Life *en) {
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      current.cell[i][j] = 0;
    }
  }
  const int ystart = en->ycoord;
  const int yend = en->ycoord + en->height;
  const int xstart = en->xcoord;
  const int xend = en->xcoord + en->width;

  int idx = 0;
  for(int i = ystart; i < yend; i++) {
    for(int j = xstart; j < xend; j++) {
      if (i < HEIGHT && j < WIDTH) {
        current.cell[i][j] = en->state[idx];
        idx++;
      } else {
        continue;
      }
    }
  }

}

void GameOfLife::update() {
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      buffer.cell[i][j] = current.cell[i][j];
    }
  }
}

int GameOfLife::nextState(int ycoord, int xcoord) {
  int  neighbours = 0;
  for (int i = ycoord - 1; i <= ycoord + 1; i++) {
    for (int j = xcoord - 1; j <= xcoord + 1; j++) {
      if (i == ycoord && j == xcoord) continue;
      if (i > -1 && i < HEIGHT && j > -1 && j < WIDTH) {
        if(buffer.cell[i][j]) {
          // std::cout << i << ":" << j << std::endl;
          neighbours++;
        }
      }
    }
  }
  uint8_t alive = buffer.cell[ycoord][xcoord];
  if (alive) {
    // std::cout << ycoord << ":" << xcoord << " = " << neighbours << std::endl;
    return neighbours > 1 && neighbours < 4;
  } else {
    return neighbours == 3;
  }
}

void GameOfLife::iterate() {
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      buffer.cell[i][j] = current.cell[i][j];
      current.cell[i][j] = 0;
    }
  }

  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      current.cell[i][j] = nextState(i, j);
    }
  }
}

void GameOfLife::print() {
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      char val = current.cell[i][j] ? 'x' : '.';
      std::cout << val;
    }
    std::cout << std::endl;
  }
}
