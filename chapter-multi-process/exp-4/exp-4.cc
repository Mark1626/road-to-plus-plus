#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/string.hpp>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>

namespace bmpi = boost::mpi;

double f(double x) { return (x <= 0) ? 0.0 : 1.0 / sqrt(x); }

class Distributor {
  const bmpi::environment env;

public:
  const bmpi::communicator world;
  const int rank;
  const int size;
  bool isMaster() { return rank == 0; }
  Distributor(int argc, char **argv)
      : env(argc, argv), world(), rank(world.rank()), size(world.size()) {}
};

int main(int argc, char **argv) {
  Distributor dist(argc, argv);
  int n;

  if (dist.isMaster()) {
    n = 1000;
  }

  double quadrile = 0.0;

  boost::mpi::broadcast(dist.world, n, 0);

  double lower_limit = 0.0;
  double upper_limit = 1.0;
  double h = (upper_limit - lower_limit) / (double)(n); // dx

  double quadrile_chunk = 0.0;
  double n_chunk = 0;

  // Divide the work for this rank
  for (int i = dist.rank + 1; i < n; i += dist.size) {
    double x = ((double)(2 * n - 2 * i + 1) * lower_limit) +
               ((double)(2 * i - 1) * upper_limit) / (double)(2 * n);
    n_chunk += 1;
    quadrile_chunk += f(x);
  }

  quadrile_chunk = quadrile_chunk * h;

  std::cout << "Quad Chunk " << quadrile_chunk << " points: " << n_chunk
            << std::endl;

  if (dist.isMaster()) {
    boost::mpi::reduce(dist.world, quadrile_chunk, quadrile,
                       std::plus<double>(), 0);
  } else {
    boost::mpi::reduce(dist.world, quadrile_chunk, std::plus<double>(), 0);
  }

  if (dist.isMaster()) {
    std::cout << "Result Quadrile " << quadrile << std::endl;
  }

  return EXIT_SUCCESS;
}
