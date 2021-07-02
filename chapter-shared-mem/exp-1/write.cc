#include <cstdio>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

#define FILE_PATH "/mark_shm"

int main() {
  int fd = shm_open(FILE_PATH, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  int res = ftruncate(fd, 256);
  void* addr = mmap(NULL, 256, PROT_WRITE, MAP_SHARED, fd, 0);

  char data[256] = "Hello World";
  size_t len = strlen(data) + 1;
  std::memcpy(addr, data, len);

printf("Sleeping\n");
  sleep(5);
  res = munmap(addr, 256);
  fd = shm_unlink(FILE_PATH);
  return EXIT_SUCCESS;
}
