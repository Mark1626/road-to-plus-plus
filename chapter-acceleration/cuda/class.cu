#include <cstdio>
#include <cstdint>

class Centroid {
public:
  double mean;
  double weight;

  __device__ Centroid(double mean, double weight)
      : mean(mean), weight(weight) {}
    
  __device__ void add(Centroid &c) {
    weight += c.weight;
    mean += c.weight * (c.mean - mean) / weight;
  }
};

__global__ void process() {
  Centroid c1(1.0, 1.0);
  Centroid c2(2.0, 1.0);

  c1.add(c2);
  printf("Mean %f\n", c1.mean);
  printf("Weight %f\n", c1.weight);
}

int main() {
  process<<<1, 1>>>();
  cudaDeviceSynchronize();
}
