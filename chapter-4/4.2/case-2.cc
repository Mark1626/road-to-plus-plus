#include <stdio.h>

int main() {
  int LIMIT = 10000;
  int count = 0;
  int found = 0;
  #pragma omp parallel default(none) shared(count, found, LIMIT)
  {

    #pragma omp for
    for (int j = 0; j < LIMIT; j++) {
      if (!found) {
        #pragma omp critical
        {
          found = 1;
          count++;
        }
        #pragma omp cancel for
      }
      if (found) {
        #pragma omp cancel for
      }
    }

    #pragma omp master
    {
      printf("Count : %d\n", count);
      count = 0;
      found = 0;
    }

    // Cancel for nested for loops
    #pragma omp for
    for (int i = 0; i < LIMIT; i++) {
      for (int j = 0; j < LIMIT; j++) {
        if (!found) {
          #pragma omp critical
          {
            found = 1;
            count++;
          }
          #pragma omp cancel for
        }
        if (found) {
          #pragma omp cancel for
        }
      }
      #pragma omp cancellation point for
    }

    #pragma omp master
    {
      printf("Count : %d\n", count);
      count = 0;
      found = 0;
    }
  }

  return 0;
}
