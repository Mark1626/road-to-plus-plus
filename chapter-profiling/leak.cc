#include <iostream>

int main() {
  int *arr = new int[10];

  delete[] arr;

  std::cout << "Testing memory leak " << arr[1];
}
