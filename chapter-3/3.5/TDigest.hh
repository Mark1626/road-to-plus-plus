/*
 *  Based on the reference implementation
 * https://github.com/derrickburns/tdigest
 */

#ifndef TDIGEST_H
#define TDIGEST_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <queue>
#include <vector>

using Value = double;
using Weight = double;
using Index = size_t;

// Max buffer size
const size_t kHigh = 40000;
class Centroid {
  Value mean_ = 0.0;
  Weight weight_ = 0.0;
  Value meansq_ = 0.0;

public:
  Centroid() : Centroid(0.0, 0.0) {}
  Centroid(Value mean) : mean_(mean), weight_(1.0), meansq_(mean * mean) {}
  Centroid(Value mean, Weight weight)
      : mean_(mean), weight_(weight), meansq_(mean * mean) {}

  // Accessors
  inline Value mean() const noexcept { return mean_; }
  inline Weight weight() const noexcept { return weight_; }
  inline Weight meansq() const noexcept { return meansq_; }

  // Centroid Add
  inline void add(const Centroid &c) {
    if (weight_ != 0.0) {
      weight_ += c.weight_;
      Value delta = c.weight_ * (c.mean_ - mean_) / weight_;
      mean_ += delta;
      meansq_ += c.meansq_;

    } else {
      weight_ = c.weight_;
      mean_ = c.mean_;
      meansq_ += c.meansq_;
    }
  }
};

struct CentroidList {
  CentroidList(const std::vector<Centroid> &s)
      : iter(s.cbegin()), end(s.cend()) {}
  std::vector<Centroid>::const_iterator iter;
  std::vector<Centroid>::const_iterator end;

  bool advance() { return ++iter != end; }
};

class CentroidListComparator {
public:
  CentroidListComparator() {}

  bool operator()(const CentroidList &left, const CentroidList &right) const {
    return left.iter->mean() > right.iter->mean();
  }
};

struct CentroidComparator {
  bool operator()(const Centroid &a, const Centroid &b) const {
    return a.mean() < b.mean();
  }
};

using CentroidListQueue =
    std::priority_queue<CentroidList, std::vector<CentroidList>,
                        CentroidListComparator>;

class TDigest {
  struct TDigestComparator {
    bool operator()(const TDigest *left, const TDigest *right) const {
      return left->totalSize() > right->totalSize();
    }
  };

  using TDigestQueue =
      std::priority_queue<const TDigest *, std::vector<const TDigest *>,
                          TDigestComparator>;

  Value compression;
  Value min = std::numeric_limits<Value>::max();
  Value max = std::numeric_limits<Value>::min();
  Index maxProcessed;
  Index maxUnprocessed;
  Value processedWeight = 0.0;
  Value unprocessedWeight = 0.0;

  inline Value mean(int i) const noexcept { return processed[i].mean(); }
  inline Value weight(int i) const noexcept { return processed[i].weight(); }

public:
  std::vector<Centroid> processed;
  std::vector<Centroid> unprocessed;
  std::vector<Weight> cumulative;

  TDigest() : TDigest(1000) {}
  TDigest(Value compression) : TDigest(compression, 0) {}
  TDigest(Value compression, Index bufferSize)
      : TDigest(compression, bufferSize, 0) {}
  TDigest(Value compression, Index unmergedSize, Index mergedSize)
      : compression(compression),
        maxProcessed(processedSize(mergedSize, compression)),
        maxUnprocessed(unprocessedSize(unmergedSize, compression)) {
    processed.reserve(maxProcessed);
    unprocessed.reserve(maxUnprocessed + 1);
  }

  size_t totalSize() const { return processed.size() + unprocessed.size(); }

  static Weight weight(std::vector<Centroid> &centroids) noexcept {
    Weight w = 0.0;
    for (auto centroid : centroids) {
      w += centroid.weight();
    }
    return w;
  }

  static inline Index processedSize(Index size, Value compression) noexcept {
    return (size == 0) ? static_cast<Index>(2 * std::ceil(compression)) : size;
  }

  static inline Index unprocessedSize(Index size, Value compression) noexcept {
    return (size == 0) ? static_cast<Index>(8 * std::ceil(compression)) : size;
  }

  inline void merge(TDigest *other) {
    std::vector<TDigest *> others{other};
    add(others.cbegin(), others.cend());
  }

  void mergeUnprocessed(const std::vector<const TDigest *> &tdigests);
  void mergeProcessed(const std::vector<const TDigest *> &tdigests);

  void add(std::vector<Centroid>::const_iterator iter,
           std::vector<Centroid>::const_iterator end);

  void add(std::vector<TDigest *>::const_iterator iter,
           std::vector<TDigest *>::const_iterator end);
  bool add(Value x);
  bool add(Value x, Weight w);

  bool haveUnprocessed() const;

  Value quantile(Value q);
  Value quantileProcessed(Value q) const;

  Value mean();
  Value rms();
  Value madfm();

  void debug();

  inline void processIfNecessary();

  bool isDirty();
  void updateCumulative();

  inline void process();
  inline void compress();

  inline Value integratedLocation(Value q) const;
  inline Value integratedQ(Value k) const;
};

Value weightedAverage(Value x1, Value w1, Value x2, Value w2);
Value weightedAverageSorted(Value x1, Value w1, Value x2, Value w2);
Value interpolate(Value x, Value x0, Value x1);
Value quantile(Value index, Value previousIndex, Value nextIndex,
               Value previousMean, Value nextMean);

#endif
