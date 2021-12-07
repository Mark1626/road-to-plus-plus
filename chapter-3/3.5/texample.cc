#include "tdigest.cc"
#include <log4cxx/log4cxx.h>
#include <log4cxx/logger.h>
#include <string>

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

void base_example(int N, double compression) {
  LOG4CXX_INFO(logger, "limit " << N << " compression " << compression);
  std::uniform_real_distribution<> reals(0.0, 1.0);
  std::random_device gen;

  TDigest digest(compression);

  // std::vector<double> values;
  for (int i = 0; i < N; i++) {
    auto value = reals(gen);
    // values.push_back(value);
    digest.add(value);
  }
  digest.compress();

  // auto m = values.begin() + values.size() / 2;
  // std::nth_element(values.begin(), m, values.end());
  // auto median = values[values.size() / 2];
  // LOG4CXX_INFO(logger, "Nth element median " << median);

  // std::sort(values.begin(), values.end());
  // auto quantile_median = quantile(0.5, values);
  // LOG4CXX_INFO(logger, "Quantile median " << quantile_median);

  // TDigest digest(100);
  // digest.add(value)
  LOG4CXX_DEBUG(logger,
               "Processed size : " << std::to_string(digest.processed.size()))
  LOG4CXX_DEBUG(logger,
               "Cumulative size : " << std::to_string(digest.cumulative.size()))

  auto digest_median = digest.quantile(0.5);
  LOG4CXX_INFO(logger, "tdigist median " << digest_median);
}

void parallel_example(int N, double compression) {
  std::uniform_real_distribution<> reals(0.0, 1.0);
  std::random_device gen;
  TDigest digest(compression);

  int n_sections = 10;
  int section_width = N / n_sections;

  std::vector<double> values;
  for (int i = 0; i < N; i++) {
    auto value = reals(gen);
    values.push_back(value);
    digest.add(value);
  }

  // std::vector<TDigest*> local_digests;
  // local_digests.resize(n_sections)

  #pragma omp parallel for num_threads(4)
  for (int section = 0; section < n_sections; section++) {
    TDigest local_digest(100);

    for (int i = 0; i < section_width; i++) {
      int idx = (section * section_width) + i;
      local_digest.add(values[idx]);
    }

    #pragma omp critical
    {
      digest.merge(&local_digest);
    }
  }
  // digest.add(local_digests.begin(), local_digests.end());
  digest.compress();

  LOG4CXX_INFO(logger, "Median " << digest.quantile(0.5));
}

#ifndef LIMIT
#define LIMIT 100000
#endif

#ifndef COMPRESSION
#define COMPRESSION 128
#endif

int main() {
  log4cxx::BasicConfigurator::configure();
  logger->setLevel(log4cxx::Level::getInfo());

  LOG4CXX_INFO(logger, "Starting main");

  parallel_example(LIMIT, COMPRESSION);

  base_example(LIMIT, COMPRESSION);

  //   std::string str = "";
  //   for (auto value : values)
  //     str += std::to_string(value) + " ";
  //   LOG4CXX_INFO(logger, str);

  LOG4CXX_INFO(logger, "End main");

  return 0;
}