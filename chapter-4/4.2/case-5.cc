#include <omp.h>
#include <stdio.h>

void thread_behaviour_in_reduction() {
  int common = 0;
  #pragma omp parallel reduction(+ : common) num_threads(2)
  {
    int tid = omp_get_thread_num() + 1;

    common += tid;

    #pragma omp barrier

    printf("common: %d\n", common);
  }

  printf("final common: %d\n", common);
}

void thread_behaviour_in_atomic() {
  int common = 0;
  #pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num() + 1;

    #pragma omp atomic
    common += tid;

    #pragma omp barrier

    printf("common: %d\n", common);
  }

  printf("final common: %d\n", common);
}

void thread_behaviour_in_critical() {
  int common = 0;
  #pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num() + 1;

    #pragma omp critical
    { common += tid; }

    #pragma omp barrier

    printf("common: %d\n", common);
  }

  printf("final common: %d\n", common);
}

int main() {
  printf("Behaviour of reduction\n");
  thread_behaviour_in_reduction();
  printf("Behaviour of atomic\n");
  thread_behaviour_in_atomic();
  printf("Behaviour of critical\n");
  thread_behaviour_in_critical();
}
