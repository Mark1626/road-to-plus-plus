#include <cstdio>
#include <omp.h>

void conditional(bool omp) {
    int t = 0;

    #pragma omp parallel if(omp) reduction(+:t)
    {
        #pragma omp atomic
        t+= 1;
    }
    printf("Threads used %d\n", t);
}

#define N 16

void fn() {
    #pragma omp parallel num_threads(2)
    {
        printf("Inner: %d\n", omp_get_num_threads());
    }
}

void nested() {
    omp_set_nested(1);
    omp_set_num_threads(1);
    #pragma omp parallel
    {
        printf("Outer: %d\n", omp_get_num_threads());
        fn();
    }
}

void cond_nested(bool omp) {
    #pragma omp parallel if(omp)
    {
        printf("Outer: %d\n", omp_get_num_threads());
        fn();
    }
}

int main(int argc, char** argv) {
    bool omp = false;
    if (argc > 1) {
        omp = true;
    }

    // conditional(omp);
    // nested();
    cond_nested(omp);
}
