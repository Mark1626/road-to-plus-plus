#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

namespace tdigest {

template <typename T> class Centroid {
public:
  T _mean;
  T _weight;
  T _meansq;
  Centroid() : Centroid(0.0, 0.0, 0.0) {}
  Centroid(T mean) : Centroid(mean, 1.0, mean * mean) {}
  Centroid(T mean, T weight) : Centroid(mean, weight, mean * mean) {}
  Centroid(T mean, T weight, T meansq)
      : _mean(mean), _weight(weight), _meansq(meansq) {}

  // Centroid(const Centroid<T>& c) = delete;

  // Centroid<T> &operator=(Centroid<T> &&o) {
  //   _mean = o._mean;
  //   _weight = o._weight;
  //   _meansq = o._meansq;
  // }

  inline T mean() const { return _mean; }
  inline T weight() const { return _weight; }
  inline T meansq() const { return _meansq; }
  inline void add(const Centroid<T> &c) {
    _weight += c._weight;
    _mean += c._weight * (c._mean - _mean) / _weight;
    _meansq += c._meansq;
  }
  inline void clear() {
    _mean = 0.0;
    _weight = 0.0;
    _meansq = 0.0;
  }
};

template <typename T> struct CentroidComparator {
  bool operator()(const Centroid<T> &a, const Centroid<T> &b) {
    return a.mean() < b.mean();
  };
};

std::ostream &operator<<(std::ostream &stream,
                         const Centroid<float> &centroid) {
  stream << "(" << centroid.mean() << ", " << centroid.weight() << ", "
         << centroid.meansq() << ")\n";
  return stream;
}

template <typename T> struct CentroidList {
  CentroidList(const std::vector<Centroid<T>> &s)
      : iter(s.cbegin()), end(s.cend()) {}
  typename std::vector<Centroid<T>>::iterator iter;
  typename std::vector<Centroid<T>>::iterator end;

  bool advance() { return ++iter != end; }
};

template <typename T> struct CentroidListComparator {
public:
  bool operator()(const CentroidList<T> &left,
                  const CentroidList<T> &right) const {
    return left.iter->mean() > right.iter->mean();
  }
};

template <typename T> class TDigest {
  T compression;
  std::vector<Centroid<T>> processed;
  std::vector<Centroid<T>> unprocessed;
  std::size_t const maxProcessed;
  std::size_t const maxUnprocessed;
  T unprocessedWeight = 0.0;
  T processedWeight = 0.0;
  T min = std::numeric_limits<T>::max();
  T max = std::numeric_limits<T>::min();
  std::vector<T> cumulative;
  Centroid<T> aggregate;

  inline T mean(int i) const noexcept { return processed[i].mean(); }
  inline T weight(int i) const noexcept { return processed[i].weight(); }

  struct Comparator {
    bool operator()(const TDigest<T> &left, const TDigest<T> &right) const {
      return left.totalSize() < right.totalSize();
    }
  };

public:
  TDigest() : TDigest(1000) {}
  TDigest(T compression)
      : compression(compression), maxProcessed(2 * std::ceil(compression)),
        maxUnprocessed(8 * std::ceil(compression)) {}

  //   TDigest<T>(TDigest<T> &o) {
  //     compression = o.compression;
  //     maxUnprocessed = o.maxUnprocessed;
  //     maxProcessed = o.maxProcessed;
  //     processedWeight = o.processedWeight;
  //     unprocessedWeight = o.unprocessedWeight;
  //     processed = std::move(o.processed);
  //     unprocessed = std::move(o.unprocessed);
  //     cumulative = std::move(o.cumulative);
  //     aggregate = std::move(o.aggregate);
  //     min = o.min;
  //     max = o.max;
  //     return *this;
  //   }

  //   // TDigest<T>& operator=(TDigest<T> &&o) {
  //   //   compression = o.compression;
  //   //   maxUnprocessed = o.maxUnprocessed;
  //   //   maxProcessed = o.maxProcessed;
  //   //   processedWeight = o.processedWeight;
  //   //   unprocessedWeight = o.unprocessedWeight;
  //   //   processed = std::move(o.processed);
  //   //   unprocessed = std::move(o.unprocessed);
  //   //   cumulative = std::move(o.cumulative);
  //   //   aggregate = std::move(o.aggregate);
  //   //   min = o.min;
  //   //   max = o.max;
  //   //   return *this;
  //   // }

  inline bool haveUnprocessed() const { return unprocessed.size() > 0; }

  inline bool overflow() {
    return processed.size() > maxProcessed ||
           unprocessed.size() > maxUnprocessed;
  }

  inline void processDigest() {
    if (haveUnprocessed() || overflow())
      process();
  }

  inline size_t totalSize() { return processed.size() + unprocessed.size(); }

  void mergeProcessed(const std::vector<TDigest<T>> &tdigests) {
    if (tdigests.size() == 0) {
      return;
    }

    // Merge N lists sorted such that the final list is sorted
    size_t total = 0;
    std::priority_queue<CentroidList<T>, std::vector<CentroidList<T>>,
                        CentroidListComparator<T>>
        queue;

    for (auto &digest : tdigests) {
      auto &clusters = digest->processed;
      auto size = clusters.size();
      if (size > 0) {
        queue.push(CentroidList<T>(clusters));
      }
    }

    // All digests have no processed elements
    if (total == 0)
      return;

    if (processed.size() > 0) {
      queue.push(CentroidList<T>(processed));
      total += processed.size();
    }

    std::vector<Centroid<T>> clusters;
    clusters.reserve(total);
    while (!queue.empty()) {
      auto top = queue.top();
      clusters.push_back(*(top.iter));
      if (top.advance()) {
        queue.push(top);
      }
    }
    processed = std::move(clusters);
  }

  void mergeUnprocessed(const std::vector<TDigest<T>> &tdigests) {
    if (tdigests.size() == 0)
      return;

    size_t total_unprocessed = unprocessed.size();
    for (auto &digest : tdigests) {
      total_unprocessed += digest->unprocessed.size();
    }

    unprocessed.reserve(total_unprocessed);
    for (auto &digest : tdigests) {
      unprocessed.insert(unprocessed.end(), digest->unprocessed.cbegin(),
                         digest->unprocessed.cend());
      unprocessedWeight += digest->unprocessedWeight;
    }
  }

  void add(TDigest<T> &other) {
    std::vector<TDigest<T>> others{other};
    add(others);
  }

  const int maxBatchSize = 40000;

  void add(std::vector<TDigest<T>> tdigests) {
    auto size = tdigests.size();
    std::priority_queue<TDigest<T>, std::vector<TDigest<T>>, Comparator> queue;

    for (auto digest : tdigests)
      queue.push(digest);

    std::vector<TDigest> batch;
    batch.reserve(size);

    size_t batchSize = 0;
    while (!queue.empty()) {
      auto top = queue.top();
      batch.push_back(top);
      queue.pop();

      batchSize += top->totalSize();
      if (batchSize >= maxBatchSize || queue.empty()) {
        mergeProcessed(batch);
        mergeUnprocessed(batch);
        processDigest();
        batch.clear();
        batchSize = 0;
      }
    }
  }

  //   void add(std::vector<Centroid<T>> centroids) {
  //     for (auto iter = centroids.begin(); iter != centroids.end();) {

  //       auto centroid = *iter;
  //       auto diff = std::distance(centroids.cbegin(), centroids.cend());

  //       // Every time we hit the maxUnprocessed limit we have to process it
  //       before
  //       // adding more clusters
  //       auto room = maxUnprocessed - unprocessed.size();

  //       auto mid = iter + std::min(diff, room);
  //       while (iter != mid)
  //         unprocessed.push_back(*(iter++));

  //       if (unprocessed.size() >= maxUnprocessed)
  //         process();
  //     }
  //   }

  bool add(T x) { return add(x, 1.0); }

  bool add(T x, T w) {
    if (std::isnan(x)) {
      return false;
    }
    Centroid<T> centroid(x, w);
    unprocessed.push_back(centroid);
    unprocessedWeight += w;
    processDigest();
    return true;
  }

    T mean() {
      processDigest();
      return aggregate.mean();
    }

    T rms() {
      processDigest();
      auto mean = aggregate.mean();
      auto weight = aggregate.weight();
      auto meansq = aggregate.meansq();
      auto rms = sqrt(meansq / weight - mean * mean);
      return rms;
    }

    T madfm() {
      processDigest();

      auto median = quantile(0.5);
      TDigest<T> diff_digest(compression);
      for (auto centroid : processed) {
        T deviation = median - centroid.mean();
        deviation = deviation < 0 ? -deviation : deviation;
        diff_digest.add(deviation, centroid.weight());
      }

      return diff_digest.quantile(0.5);
    }

  void debug() {
    std::cout << "Unprocessed" << std::endl;
    for (auto centroid : unprocessed)
      std::cout << centroid;
    std::cout << "Processed" << std::endl;
    for (auto centroid : processed)
      std::cout << centroid;
  }

  static inline T weightedAverage(T x1, T w1, T x2, T w2) {
    if (x1 > x2) {
      std::swap(x1, x2);
      std::swap(w1, w2);
    }
    const T x = (x1 * w1 + x2 * w2) / (w1 + w2);
    return std::max(x1, std::min(x, x2));
  }

  T quantile(T q) {
    processDigest();
    if (q < 0 || q > 1) {
      std::cerr << "q is not within range of 0 and 1";
      return NAN;
    }

    if (processed.size() == 0) {
      return NAN;
    } else if (processed.size() == 1) {
      return mean(0);
    }

    auto n = processed.size();
    const auto index = q * processedWeight;

    if (index <= weight(0) / 2) {
      return min + 2.0 * index / weight(0) * (mean(0) - min);
    }

    typename std::vector<T>::const_iterator iter = std::lower_bound(cumulative.begin(), cumulative.end(), index);

    // auto iter = cumulative.cbegin();
    // for (; iter != cumulative.cend(); iter++) {
    //   if (*iter > index) {
    //     break;
    //   }
    // }

    if (iter + 1 != cumulative.cend()) {
      auto i = std::distance(cumulative.cbegin(), iter);
      auto z1 = index - *(iter - 1);
      auto z2 = *(iter)-index;
      return weightedAverage(mean(i - 1), z2, mean(i), z1);
    }

    auto z1 = index - processedWeight - weight(n - 1) / 2.0;
    auto z2 = weight(n - 1) / 2 - z1;

    return weightedAverage(mean(n - 1), z1, max, z2);
  }

  // $$ c * (2sin^{-1}(2q - 1) + \pi / 2) / \2pi$$
  inline T integratedLocation(T q) const {
    return compression * (std::asin(2.0 * q - 1.0) + M_PI / 2) / M_PI;
  }

  // $$ (sin(min(k, c) * \pi / c - \pi / 2) + 1) / 2 $$
  inline T integratedQ(T k) const {
    return (std::sin(std::min(k, compression) * M_PI / compression - M_PI / 2) +
            1) /
           2;
  }

  inline void process() {
    CentroidComparator<T> cc;
    std::sort(unprocessed.begin(), unprocessed.end(), cc);

    auto count = unprocessed.size();
    unprocessed.insert(unprocessed.end(), processed.cbegin(), processed.cend());
    std::inplace_merge(unprocessed.begin(), unprocessed.begin() + count,
                       unprocessed.end(), cc);

    processedWeight += unprocessedWeight;
    unprocessedWeight = 0;
    processed.clear();
    processed.push_back(unprocessed[0]);
    T w = unprocessed[0].weight();
    T qlimit = processedWeight * integratedQ(1.0);

    auto end = unprocessed.end();
    for (auto iter = unprocessed.cbegin() + 1; iter < end; iter++) {
      auto &centroid = *iter;
      T projected = w + centroid.weight();
      if (projected <= qlimit) {
        w = projected;
        (processed.end() - 1)->add(centroid);
      } else {
        auto k1 = integratedLocation(w / processedWeight);
        qlimit = processedWeight * integratedQ(k1 + 1.0);
        w += centroid.weight();
        processed.emplace_back(centroid);
      }
    }
    unprocessed.clear();
    updateCumulative();
  }

  void updateCumulative() {
    // Update 0.0 and 1.0 quantile
    min = std::min(min, mean(0));
    max = std::max(max, (processed.cend() - 1)->mean());

    // Update cumulative
    const auto n = processed.size();
    cumulative.clear();
    cumulative.reserve(n + 1);
    T previous = 0.0;
    for (size_t i = 0; i < n; i++) {
      T w = weight(i);
      cumulative.push_back(previous + w / 2.0);
      previous += w;
    }
    cumulative.push_back(previous);

    // Update aggregate
    aggregate.clear();
    for (auto centroid : processed) {
      aggregate.add(centroid);
    }
  }
};
} // namespace tdigest
