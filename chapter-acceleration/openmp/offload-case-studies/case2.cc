#include <cstdlib>
#include <cstdio>

struct S {
  int N;
  float a;
  float b;
  float *points;
  S(int N, float a, float b) : N(N), a(a), b(b), points(new float[N]) {
    float x = 5.0;
    srand(10);
    for (int i = 0; i < N; i++) {
      auto val = (float)std::rand() / (float)(RAND_MAX / x);
      points[i] = val;
    }
  }
  ~S() { delete[] points; }
};

#pragma omp declare target
void saxpy(S *s) {
  for (int i = 0; i < s->N; i++) {
    s->points[i] = s->points[i] * s->a + s->b;
  }
}
#pragma omp end declare target

void test_offload_struct(int N) {
  S s(N, 2.0, 1.0);

#pragma omp target map(to : s.N, s.a, s.b) map(tofrom : s.points[:N])
  saxpy(&s);

  printf("%f\n", s.points[0]);
}
