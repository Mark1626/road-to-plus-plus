#include "lib.cuh"
#include <cstdio>

int main() {
    int N = 1000000;
    float *a = new float[N];
    float *b = new float[N];

    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = i + 1;
    }

    float *c = new float[N];
    saxpy(N, a, b, c);

    printf("%f %f %f\n", c[1], c[5], c[200]);

    delete [] c;
    delete [] b;
    delete [] a;
}
