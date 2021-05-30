#pragma once

#include <cstdlib>

class Life {
public:
  uint8_t xcoord;
  uint8_t ycoord;
  uint8_t height;
  uint8_t width;
  uint8_t *state;
  Life(uint8_t xcoord, uint8_t ycoord, uint8_t height, uint8_t width,
         uint8_t* state)
      : xcoord(xcoord), ycoord(ycoord), height(height), width(width),
        state(state){};
};

// Spaceship
class Glider : public Life {
  private:
    uint8_t state[9] = {
      0, 1, 0,
      0, 0, 1,
      1, 1, 1,
    };
  public:
  Glider(uint8_t xpos, uint8_t ypos) : Life(xpos, ypos, 3, 3, state) { };
};

// Still life
class Block : public Life {
  private:
    uint8_t state[4] = {
    1, 1,
    1, 1,
  };
  public:
  Block(uint8_t xpos, uint8_t ypos) : Life(xpos, ypos, 2, 2, state) { };
};

class BeeHive : public Life {
  private:
    uint8_t state[12] = {
    0, 1, 1, 0,
    1, 0, 0, 1,
    0, 1, 1, 0,
  };
  public:
  BeeHive(uint8_t xpos, uint8_t ypos) : Life(xpos, ypos, 3, 4, state) { };
};

class Loaf : public Life {
  private:
    uint8_t state[16] = {
    0, 1, 1, 0,
    1, 0, 0, 1,
    0, 1, 0, 1,
    0, 0, 1, 0,
  };
  public:
  Loaf(uint8_t xpos, uint8_t ypos) : Life(xpos, ypos, 4, 4, state) { };
};

// Oscillator
class Oscillator : public Life {
  private:
    uint8_t state[9] = {
    0, 1, 0,
    0, 1, 0,
    0, 1, 0
  };
  public:
  Oscillator(uint8_t xpos, uint8_t ypos) : Life(xpos, ypos, 3, 3, state) { };
};

// Glider Gun
class GliderGun : public Life {
  private:
    uint8_t state[324] = {
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
      0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
      1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };
  public:
  GliderGun(uint8_t xpos, uint8_t ypos) : Life(xpos, ypos, 9, 36, state) { };
};
