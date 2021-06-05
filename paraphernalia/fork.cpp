#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>

int main() {
  std::printf("Parent %d\n", getpid());

  for (int i = 0; i < 4; ++i) {
    pid_t pid = fork();
    if (pid) {
      std::printf("parent %d\n", getpid());
      break;
    }
    std::printf("Child %d\n", getpid());
  }

  std::printf("Hello\n");

  wait(NULL);
  return EXIT_SUCCESS;
}
