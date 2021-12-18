/*
 *  Based on the reference implementation
 * https://github.com/derrickburns/tdigest
 */

#include "TDigest.hh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <queue>
#include <vector>

std::ostream &operator<<(std::ostream &stream, const Centroid &centroid) {
  stream << "(" << centroid.mean() << ", " << centroid.weight() << ", "
         << centroid.meansq() << ")\n";
  return stream;
}

void TDigest::mergeUnprocessed(const std::vector<const TDigest *> &tdigests) {
  if (tdigests.size() == 0)
    return;

  size_t total = unprocessed.size();
  for (auto &td : tdigests) {
    total += td->unprocessed.size();
  }

  unprocessed.reserve(total);
  for (auto &td : tdigests) {
    unprocessed.insert(unprocessed.end(), td->unprocessed.cbegin(),
                       td->unprocessed.cend());
    unprocessedWeight += td->unprocessedWeight;
  }
}

void TDigest::mergeProcessed(const std::vector<const TDigest *> &tdigests) {
  if (tdigests.size() == 0)
    return;

  size_t total = 0;
  CentroidListQueue pq(CentroidListComparator{});
  for (auto &td : tdigests) {
    auto &sorted = td->processed;
    auto size = sorted.size();
    if (size > 0) {
      pq.push(CentroidList(sorted));
      total += size;
      processedWeight += td->processedWeight;
    }
  }
  if (total == 0)
    return;

  if (processed.size() > 0) {
    pq.push(CentroidList(processed));
    total += processed.size();
  }

  std::vector<Centroid> sorted;
  // LOG4CXX_DEBUG(logger, "total " << total);
  sorted.reserve(total);

  while (!pq.empty()) {
    auto best = pq.top();
    pq.pop();
    sorted.push_back(*(best.iter));
    if (best.advance())
      pq.push(best);
  }
  processed = std::move(sorted);
  if (processed.size() > 0) {
    min = std::min(min, processed[0].mean());
    max = std::max(max, (processed.cend() - 1)->mean());
  }
}

// inline void add(std::vector<const TDigest *> diges
void TDigest::add(std::vector<Centroid>::const_iterator iter,
                  std::vector<Centroid>::const_iterator end) {
  while (iter != end) {
    const size_t diff = std::distance(iter, end);
    const size_t room = maxUnprocessed - unprocessed.size();
    auto mid = iter + std::min(diff, room);
    while (iter != mid)
      unprocessed.push_back(*(iter++));

    if (unprocessed.size() >= maxUnprocessed) {
      process();
    }
  }
}

void TDigest::add(std::vector<TDigest *>::const_iterator iter,
                  std::vector<TDigest *>::const_iterator end) {

  if (iter != end) {
    auto size = std::distance(iter, end);
    TDigestQueue queue;

    for (; iter != end; iter++) {
      queue.push((*iter));
    }

    std::vector<const TDigest *> batch;
    batch.reserve(size);

    size_t totalSize = 0;
    while (!queue.empty()) {
      auto top = queue.top();
      batch.push_back(top);
      queue.pop();
      totalSize += top->totalSize();
      if (totalSize >= kHigh || queue.empty()) {
        mergeProcessed(batch);
        mergeUnprocessed(batch);
        processIfNecessary();
        batch.clear();
        totalSize = 0;
      }
    }
    updateCumulative();
  }
}

bool TDigest::add(Value x) { return add(x, 1); }

bool TDigest::add(Value x, Weight w) {
  if (std::isnan(x)) {
    return false;
  }
  unprocessed.push_back(Centroid(x, w));
  unprocessedWeight += w;
  processIfNecessary();
  return true;
}

bool TDigest::haveUnprocessed() const { return unprocessed.size() > 0; }

Value TDigest::quantile(Value q) {
  if (haveUnprocessed() || isDirty())
    process();
  return quantileProcessed(q);
}

Value TDigest::quantileProcessed(Value q) const {
  if (q < 0 || q > 1) {
    std::cerr << "q is not within range of 0 and 1";
    return NAN;
  }

  if (processed.size() == 0) {
    std::cerr << "Data is empty";
    return NAN;
  } else if (processed.size() == 1) {
    return mean(0);
  }

  auto n = processed.size();
  const auto index = q * processedWeight;

  if (index <= weight(0) / 2.0) {
    //   LOG4CXX_INFO(logger, "Boundary condition");
    return min + 2.0 * index / weight(0) * (mean(0) - min);
  }

  auto iter = std::lower_bound(cumulative.cbegin(), cumulative.cend(), index);
  if (iter + 1 != cumulative.cend()) {
    auto i = std::distance(cumulative.cbegin(), iter);
    auto z1 = index - *(iter - 1);
    auto z2 = *(iter)-index;
    //   LOG4CXX_DEBUG(logger, "x2 " << mean(i - 1) << " z2 " << z2
    //                               << " index: " << index << " x1: " <<
    //                               mean(i)
    //                               << " z1: " << z1);
    return weightedAverage(mean(i - 1), z2, mean(i), z1);
  }

  auto z1 = index - processedWeight - weight(n - 1) / 2.0;
  auto z2 = weight(n - 1) / 2 - z1;

  // LOG4CXX_DEBUG(logger, "Processing quantile: "
  //                           << "z2 " << z2 << " index: " << index
  //                           << " z1: " << z1);

  return weightedAverage(mean(n - 1), z1, max, z2);
}

Value TDigest::mean() {
  if (haveUnprocessed() || isDirty())
    process();
  Centroid c;
  for (auto centroid : processed)
    c.add(centroid);
  return c.mean();
}

Value TDigest::rms() {
  if (haveUnprocessed() || isDirty())
    process();
  Centroid c;
  for (auto centroid : processed)
    c.add(centroid);

  auto mean = c.mean();
  auto weight = c.weight();
  auto meansq = c.meansq();

  auto rms = sqrt(meansq / weight - mean * mean);

  return rms;
}

Value TDigest::madfm() {
  auto median = quantile(0.5);

  TDigest dfm_digest(compression);

  for (auto centroid : processed) {
    double deviation = median - centroid.mean();
    deviation = deviation < 0 ? -deviation : deviation;
    dfm_digest.add(deviation, centroid.weight());
  }

  return dfm_digest.quantile(0.5);
}

void TDigest::debug() {
  std::cout << "Unprocessed" << std::endl;
  for (auto centroid : unprocessed)
    std::cout << centroid;

  std::cout << "Processed" << std::endl;
  for (auto centroid : processed)
    std::cout << centroid;
}

inline void TDigest::processIfNecessary() {
  if (isDirty()) {
    process();
  }
}

bool TDigest::isDirty() {
  return processed.size() > maxProcessed || unprocessed.size() > maxUnprocessed;
}

void TDigest::updateCumulative() {
  const auto n = processed.size();
  // LOG4CXX_DEBUG(logger, "Updating cumulative size : " << std::to_string(n))
  cumulative.clear();
  cumulative.reserve(n + 1);
  auto previous = 0.0;
  for (size_t i = 0; i < n; i++) {
    auto current = weight(i);
    auto halfCurrent = current / 2.0;
    cumulative.push_back(previous + halfCurrent);
    previous += current;
  }
  cumulative.push_back(previous);
}

inline void TDigest::process() {
  CentroidComparator cc;
  std::sort(unprocessed.begin(), unprocessed.end(), cc);

  auto count = unprocessed.size();
  unprocessed.insert(unprocessed.end(), processed.cbegin(), processed.cend());
  std::inplace_merge(unprocessed.begin(), unprocessed.begin() + count,
                     unprocessed.end(), cc);

  processedWeight += unprocessedWeight;
  unprocessedWeight = 0;
  processed.clear();

  processed.push_back(unprocessed[0]);
  Weight w = unprocessed[0].weight();
  Weight wLimit = processedWeight * integratedQ(1.0);

  auto end = unprocessed.end();
  for (auto iter = unprocessed.cbegin() + 1; iter < end; iter++) {
    auto &centroid = *iter;
    Weight projected = w + centroid.weight();
    if (projected <= wLimit) {
      w = projected;
      (processed.end() - 1)->add(centroid);
    } else {
      auto k1 = integratedLocation(w / processedWeight);
      wLimit = processedWeight * integratedQ(k1 + 1.0);
      w += centroid.weight();
      processed.emplace_back(centroid);
    }
  }
  unprocessed.clear();
  min = std::min(min, processed[0].mean());
  max = std::max(max, (processed.cend() - 1)->mean());
  // LOG4CXX_DEBUG(logger, "new min " << min);
  // LOG4CXX_DEBUG(logger, "new max " << max);
  updateCumulative();
}

inline void TDigest::compress() { process(); }

// $$ c * (sin^{-1}(2q - 1) + \pi / 2) / \pi$$
inline Value TDigest::integratedLocation(Value q) const {
  return compression * (std::asin(2.0 * q - 1.0) + M_PI / 2) / M_PI;
}

// $$ (sin(min(k, c) * \pi / c - \pi / 2) + 1) / 2 $$
inline Value TDigest::integratedQ(Value k) const {
  return (std::sin(std::min(k, compression) * M_PI / compression - M_PI / 2) +
          1) /
         2;
}

Value weightedAverage(Value x1, Value w1, Value x2, Value w2) {
  return (x1 <= x2) ? weightedAverageSorted(x1, w1, x2, w2)
                    : weightedAverageSorted(x2, w2, x1, w1);
}

Value weightedAverageSorted(Value x1, Value w1, Value x2, Value w2) {
  const Value x = (x1 * w1 + x2 * w2) / (w1 + w2);
  // LOG4CXX_DEBUG(logger,
  //               "Using weighted average " << x1 << " " << x2 << " " << x);
  return std::max(x1, std::min(x, x2));
}

Value interpolate(Value x, Value x0, Value x1) { return (x - x0) / (x1 - x0); }

Value quantile(Value index, Value previousIndex, Value nextIndex,
               Value previousMean, Value nextMean) {
  const auto delta = nextIndex - previousIndex;
  const auto previousWeight = (nextIndex - index) / delta;
  const auto nextWeight = (index - previousIndex) / delta;
  return previousMean * previousWeight + nextMean * nextWeight;
}
