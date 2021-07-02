#include <cstdlib>
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

#define FILE_PATH "/mark_shm"

int main() {
  int fd = shm_open(FILE_PATH, O_RDONLY, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    std::perror("open");
    return 10;
  }
  std::printf("Reading from shared map\n");

  char data[256];
  void* addr = mmap(NULL, 256, PROT_READ, MAP_SHARED, fd, 0);
  memcpy(data, addr, 256);

  printf("Read from %s data: %s\n", FILE_PATH, data);

  return EXIT_SUCCESS;
}
