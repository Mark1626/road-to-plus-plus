#include <cstdio>
#include <cstdlib>
#include <thread>

// When compiling mention standard
void worker(int a) {
  printf("Hello from thread %d\n", a);
}

int main() {
  std::thread th1(worker, 1);
  std::thread th2(worker, 2);

  th1.join();
  th2.join();
  return EXIT_SUCCESS;
}
