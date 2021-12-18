#include "TDigest.hh"
#include <algorithm>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <log4cxx/log4cxx.h>
#include <log4cxx/logger.h>
#include <omp.h>
#include <string>
// #include <boost/mpi/environment.hpp>
// #include <boost/mpi/communicator.hpp>

// log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("tdigest-main"));

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

std::vector<double> deserialize(int N, std::string filename = "data.dat") {
  std::vector<double> restore_values;
  std::ifstream ifs(filename);

  if (ifs.is_open()) {
    boost::archive::text_iarchive ia(ifs);

    for (int i = 0; i < N; i++) {
      double value;
      ia >> value;
      restore_values.push_back(value);
    }
  } else {
    throw std::runtime_error("Unable to open file");
  }

  return restore_values;
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
  TDigest digest(compression);

  std::vector<double> values = deserialize(N);

  for (int i = 0; i < N; i++) {
    digest.add(values[i]);
  }
  digest.compress();

  auto digest_median = digest.quantile(0.5);
  TDigest dfm_digest(compression);

  for (auto centroid : digest.processed) {
    double deviation = digest_median - centroid.mean();
    deviation = deviation < 0 ? -deviation : deviation;
    dfm_digest.add(deviation, centroid.weight());
  }

  auto digest_madfm = dfm_digest.quantile(0.5);

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
  TDigest digest(compression);

  int n_sections = num_threads;
  int section_width = N / n_sections;

  std::vector<double> values = deserialize(N);

  #pragma omp parallel for num_threads(num_threads)
  for (int section = 0; section < n_sections; section++) {
    TDigest local_digest(compression);

    for (int i = 0; i < section_width; i++) {
      int idx = (section * section_width) + i;
      local_digest.add(values[idx]);
    }

  #pragma omp critical
    { digest.merge(&local_digest); }
  }
  digest.compress();

  LOG4CXX_INFO(logger, "tdigest parallel MEDIAN: " << digest.quantile(0.5));
}

/*
  Benchmark doesn't show an increase in performance, this is because processing
  is only taking place in digest.quantile
*/
void parallel_example_optimized_1(int N, double compression,
                                  const int num_threads) {
  std::vector<double> values = deserialize(N);
  TDigest digest(compression);

  int n_sections = num_threads;
  int section_width = N / n_sections;

  std::vector<TDigest *> local_digests(num_threads);

  #pragma omp parallel num_threads(num_threads) shared(values, local_digests)
  {
    TDigest *local_digest = new TDigest(compression);
    int tid = omp_get_thread_num();

    for (int i = 0; i < section_width; i++) {
      int idx = (tid * section_width) + i;
      local_digest->add(values[idx]);
    }

    local_digests[tid] = local_digest;
  }

  digest.add(local_digests.begin(), local_digests.end());

  auto digest_median = digest.quantile(0.5);
  TDigest dfm_digest(compression);

  Centroid c;
  for (auto centroid : digest.processed) {
    c.add(centroid);
  }
  auto mean = c.mean();
  // mean = mean / weight;

  for (auto centroid : digest.processed) {
    double deviation = digest_median - centroid.mean();
    deviation = deviation < 0 ? -deviation : deviation;
    dfm_digest.add(deviation);
    // mean += centroid.mean();
  }

  auto digest_madfm = dfm_digest.quantile(0.5);

  LOG4CXX_INFO(logger, "tdigest parallel optimized 1 MEAN: " << mean << " MEDIAN: "
                           << digest_median << " MADFM: " << digest_madfm);
}

/*
  Reducing memory shared with threads. This did not give any improvement,
  probable because of the increased IO bandwidth
*/
void parallel_example_optimized_2(int N, double compression,
                                  const int num_threads) {
  TDigest digest(compression);

  int n_sections = num_threads;
  int section_width = N / n_sections;

  std::vector<TDigest *> local_digests(num_threads);

#pragma omp parallel num_threads(num_threads) shared(local_digests)
  {
    int tid = omp_get_thread_num();
    std::string filename = "data_" + std::to_string(tid) + ".dat";
    std::vector<double> values = deserialize(section_width, filename);
    TDigest *local_digest = new TDigest(compression);

    for (int i = 0; i < section_width; i++) {
      local_digest->add(values[i]);
    }

    local_digests[tid] = local_digest;
  }

  digest.add(local_digests.begin(), local_digests.end());
  LOG4CXX_INFO(logger,
               "tdigest parallel optimized 2 MEDIAN: " << digest.quantile(0.5));
}

// void base_openmpi(int N, double compression, const int num_processes, int
// argc, char**argv) {
//   MPI_Init(&argc, &argv);
//   MPI_Finalise();

// }

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
  // base_example(LIMIT, COMPRESSION);
  // parallel_example(LIMIT, COMPRESSION, NUM_THREADS);
  parallel_example_optimized_1(LIMIT, COMPRESSION, NUM_THREADS);
  // parallel_example_optimized_2(LIMIT, COMPRESSION, NUM_THREADS);

  //   std::string str = "";
  //   for (auto value : values)
  //     str += std::to_string(value) + " ";
  //   LOG4CXX_INFO(logger, str);

  LOG4CXX_INFO(logger, "End main");

  return 0;
}