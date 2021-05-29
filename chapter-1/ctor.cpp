#include <iostream>

class Test {
public:
  int n;
  char c;
  // Default ctor
  Test() { this->n = 0; this->c = 'o'; }
  // Parameterised ctor with initializer list
  Test(int n) : n(n), c{'c'} {}
  // Copy ctor with initializer list
  Test(const Test &t) : n(t.n) {}
  // Move ctor
  Test(Test&& t) = default;
};

int main() {
  Test *t1 = new Test();
  std::cout << "Test 1: " << t1->n << std::endl;

  Test t2(10);
  std::cout << "Test 2: " << t2.n << std::endl;

  // Copy constructor on value referenced by pointer
  Test t3(*t1);
  std::cout << "Test 3: " << t3.n << std::endl;

  Test t4(t2);
  std::cout << "Test 4: " << t4.n << std::endl;

  return 0;
}
