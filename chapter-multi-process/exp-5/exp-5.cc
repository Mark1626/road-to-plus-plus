#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename T> class Centroid {
public:
  T _mean;
  T _weight;
  T _meansq;
  Centroid() : Centroid(0.0, 0.0, 0.0) {}
  Centroid(T mean) : Centroid(mean, 1.0, mean * mean) {}
  Centroid(T mean, T weight, T meansq)
      : _mean(mean), _weight(weight), _meansq(meansq) {}

  inline T mean() const { return _mean; }
  inline T weight() const  { return _weight; }
  inline T meansq() const { return _meansq; }
  inline void add(const Centroid<T> &c) {
    _weight += c._weight;
    _mean += c._weight * (c._mean - _mean) / _weight;
    _meansq += c._meansq;
  }
};

std::ostream &operator<<(std::ostream &stream, const Centroid<float> &centroid) {
  stream << "(" << centroid.mean() << ", " << centroid.weight() << ", "
         << centroid.meansq() << ")\n";
  return stream;
}

namespace boost {
namespace serialization {
template <class A>
void serialize(A &ar, Centroid<float> &c, const unsigned int version) {
  ar &c._mean;
  ar &c._weight;
  ar &c._meansq;
}
} // namespace serialization
} // namespace boost

namespace bmpi = boost::mpi;

class Distributor {
  const bmpi::environment env;

public:
  const bmpi::communicator world;
  const int rank;
  const int size;
  inline bool isMaster() { return rank == 0; }
  Distributor(int argc, char **argv)
      : env(argc, argv), world(), rank(world.rank()), size(world.size()) {}
};

int main(int argc, char **argv) {
  Distributor dist(argc, argv);

  Centroid<float> centroid(dist.rank * 10.0);

  if (dist.isMaster()) {
    std::vector<Centroid<float>> list;
    bmpi::gather(dist.world, centroid, list, 0);
    for (auto c : list) {
      std::cout << c;
    }
  } else {
    bmpi::gather(dist.world, centroid, 0);
  }
}
