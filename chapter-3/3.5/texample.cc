#include "TDigest.hh"
#include <algorithm>
// #include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <log4cxx/log4cxx.h>
#include <log4cxx/logger.h>
#include <omp.h>
#include <string>
// #include <boost/mpi/environment.hpp>
// #include <boost/mpi/communicator.hpp>

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("tdigest-main"));

using namespace tdigest;

static double quantile(const double q, const std::vector<double> &values) {
  double q1;
  if (values.size() == 0) {
    q1 = NAN;
  } else if (q == 1 || values.size() == 1) {
    q1 = values[values.size() - 1];
  } else {
    auto index = q * values.size();
    if (index < 0.5) {
      q1 = values[0];
    } else if (values.size() - index < 0.5) {
      q1 = values[values.size() - 1];
    } else {
      index -= 0.5;
      const int intIndex = static_cast<int>(index);
      q1 = values[intIndex + 1] * (index - intIndex) +
           values[intIndex] * (intIndex + 1 - index);
    }
  }
  return q1;
}

std::vector<double> deserialize(int N) {
  double a = 5.0;
  std::vector<double> values;
  for (int i = 0; i < N; i++) {
    auto val = (double)std::rand() / (double)(RAND_MAX / a);
    values.push_back(val);
  }
  return values;
}

void reference(int N) {
  std::vector<double> values = deserialize(N);
  // std::sort(values.begin(), values.end());

  auto mean = 0.0;
  for (int i = 0; i < values.size(); i++)
    mean += values[i];
  mean = mean / values.size();

  auto mid = values.begin() + values.size() / 2;

  std::nth_element(values.begin(), mid, values.end());
  auto median = values[values.size() / 2];

  std::vector<double> deviation(values);
  for (size_t i = 0; i < deviation.size(); i++) {
    double val = deviation[i] - median;
    deviation[i] = val < 0 ? -val : val;
  }

  auto deviation_mid = deviation.begin() + deviation.size() / 2;

  std::nth_element(deviation.begin(), deviation_mid, deviation.end());
  auto madfm = deviation[deviation.size() / 2];
  LOG4CXX_INFO(logger, "Reference MEDIAN: " << median << " MEAN: " << mean << " MADFM: " << madfm);
}

void base_example(int N, double compression) {
  TDigest<double> digest(compression);

  std::vector<double> values = deserialize(N);

  for (int i = 0; i < N; i++) {
    digest.add(values[i]);
  }

  auto digest_median = digest.quantile(0.5);
  auto digest_madfm = digest.madfm();

  //   valuesSketch.centroids().forEach(centroid -> {
  //     final double deviation = Math.abs(approximateMedian - centroid.mean());
  //     approximatedDeviationsSketch.add(deviation, centroid.count());
  // });

  LOG4CXX_INFO(logger, "tdigist MEDIAN: " << digest_median
                                          << " MADFM: " << digest_madfm);
}

/*
  Base parallel example.
  This is not the best way to merge digests, merging is done one by one in a
  critical section. However the ability to parallelize computation is seen
  clearly here
*/
void parallel_example(int N, double compression, const int num_threads) {
  TDigest<double> digest(compression);

  int n_sections = num_threads;
  int section_width = N / n_sections;

  std::vector<double> values = deserialize(N);

  #pragma omp parallel for num_threads(num_threads)
  for (int section = 0; section < n_sections; section++) {
    TDigest<double> local_digest(compression);

    for (int i = 0; i < section_width; i++) {
      int idx = (section * section_width) + i;
      local_digest.add(values[idx]);
    }

  #pragma omp critical
    { digest.add(local_digest); }
  }

  LOG4CXX_INFO(logger, "tdigest parallel MEDIAN: " << digest.quantile(0.5));
}

#ifndef LIMIT
#define LIMIT 100000
#endif

#ifndef COMPRESSION
#define COMPRESSION 128
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

int main() {
  log4cxx::BasicConfigurator::configure();
  logger->setLevel(log4cxx::Level::getInfo());

  LOG4CXX_INFO(logger, "Starting main");

  LOG4CXX_INFO(logger, "limit " << LIMIT << " compression " << COMPRESSION
                                << " threads: " << NUM_THREADS);

  reference(LIMIT);
  base_example(LIMIT, COMPRESSION);
  parallel_example(LIMIT, COMPRESSION, NUM_THREADS);

  LOG4CXX_INFO(logger, "End main");

  return 0;
}