#include <pthread.h>
#include <cstdio>

void* worker(void* arg) {
  printf("Hello World from %d \n", pthread_main_np());
  return NULL;
}

int main() {
  pthread_t thread;
  pthread_create(&thread, NULL, worker, NULL);

  pthread_join(thread, NULL);
  return 0;
}
