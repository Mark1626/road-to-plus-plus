#include <iostream>

struct A {
    ~A() {
          std::cout << "A destruct\n";
            }
};

struct B {
    ~B() {
          std::cout << "B destruct\n";
            }
};

int main() {
    A a;
      B b;
}

