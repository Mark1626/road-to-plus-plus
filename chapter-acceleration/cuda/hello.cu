#include <stdio.h>
#include <unistd.h>

__global__ void hello() {
  printf("GPU: Hello World\n");
}

int main() {
  hello <<< 1, 4 >>>();
  printf("CPU: Hello world\n");

  printf("Making up for the latency with sleep\n");
  sleep(2);
}

