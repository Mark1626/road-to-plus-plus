#include <iostream>

class Test {
  int n;
  public:
  Test(int n) : n(n) {};
};

int main(int, char**) {
  auto msg = "Hello World";
  std::cout << msg << std::endl;
  return 1;
}
