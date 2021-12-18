## Compiling

```
c++ -o texample texample.cc `pkg-config --libs liblog4cxx`
c++ -o test_tdigest test_tdigest.cc `pkg-config --libs liblog4cxx cppunit`
c++ -o tbench tbench.cc `pkg-config --libs liblog4cxx benchmark`

```

## Case Studies

### Microsoft

- [Delivering 45x faster percentiles using Postgres, Citus, & t-digest](https://techcommunity.microsoft.com/t5/azure-database-for-postgresql/diary-of-an-engineer-delivering-45x-faster-percentiles-using/ba-p/1685102)

**Notes:**
  - Use case highlights t-digest used in a system with 100TB with 10TB/hr data ingestion

### Elasticsearch

[elasticsearch tdigest Median Absolute Deviation](https://github.com/elastic/elasticsearch/blob/fd185e4661ee61a99eeefe94d781839893c8bba2/server/src/main/java/org/elasticsearch/search/aggregations/metrics/InternalMedianAbsoluteDeviation.java#L30-L36)

**Notes:**
  - Data ingested once is used for median and MADFM


## Benchmark result

```
2021-12-07T15:09:17+05:30
Running ./tbench
Run on (16 X 4850.19 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 512 KiB (x8)
  L3 Unified 32768 KiB (x1)
----------------------------------------------------------------------------------------------
Benchmark                                                    Time             CPU   Iterations
----------------------------------------------------------------------------------------------
Benchmark_Sort_Quantile/1024                             0.004 ms        0.004 ms       192651
Benchmark_Sort_Quantile/4096                             0.018 ms        0.018 ms        39952
Benchmark_Sort_Quantile/32768                            0.172 ms        0.172 ms         4039
Benchmark_Sort_Quantile/262144                            1.66 ms         1.66 ms          398
Benchmark_Sort_Quantile/2097152                           18.9 ms         18.9 ms           32
Benchmark_Sort_Quantile/16777216                          1233 ms         1233 ms            1
log4cxx: No appender could be found for logger (tdigest-main).
log4cxx: Please initialize the log4cxx system properly.
Benchmark_Tdigest_Quantile/1024/128                      0.024 ms        0.024 ms        28950
Benchmark_Tdigest_Quantile/4096/128                      0.089 ms        0.089 ms         7474
Benchmark_Tdigest_Quantile/32768/128                      1.63 ms         1.63 ms          429
Benchmark_Tdigest_Quantile/262144/128                     13.1 ms         13.1 ms           53
Benchmark_Tdigest_Quantile/2097152/128                     105 ms          105 ms            7
Benchmark_Tdigest_Quantile/16777216/128                    840 ms          840 ms            1
Benchmark_Tdigest_Quantile/1024/256                      0.027 ms        0.027 ms        25882
Benchmark_Tdigest_Quantile/4096/256                      0.088 ms        0.088 ms         7385
Benchmark_Tdigest_Quantile/32768/256                      1.69 ms         1.69 ms          414
Benchmark_Tdigest_Quantile/262144/256                     13.6 ms         13.6 ms           52
Benchmark_Tdigest_Quantile/2097152/256                     109 ms          109 ms            6
Benchmark_Tdigest_Quantile/16777216/256                    870 ms          870 ms            1
Benchmark_Parallel_Tdigest_Quantile/1024/128/4           0.048 ms        0.048 ms        14705
Benchmark_Parallel_Tdigest_Quantile/4096/128/4           0.144 ms        0.144 ms         4778
Benchmark_Parallel_Tdigest_Quantile/32768/128/4          0.556 ms        0.556 ms         1262
Benchmark_Parallel_Tdigest_Quantile/262144/128/4          3.51 ms         3.51 ms          200
Benchmark_Parallel_Tdigest_Quantile/2097152/128/4         26.8 ms         26.8 ms           26
Benchmark_Parallel_Tdigest_Quantile/16777216/128/4         214 ms          214 ms            3
Benchmark_Parallel_Tdigest_Quantile/1024/256/4           0.050 ms        0.050 ms        13651
Benchmark_Parallel_Tdigest_Quantile/4096/256/4           0.182 ms        0.182 ms         3734
Benchmark_Parallel_Tdigest_Quantile/32768/256/4          0.740 ms        0.739 ms          949
Benchmark_Parallel_Tdigest_Quantile/262144/256/4          3.80 ms         3.80 ms          185
Benchmark_Parallel_Tdigest_Quantile/2097152/256/4         28.0 ms         28.0 ms           25
Benchmark_Parallel_Tdigest_Quantile/16777216/256/4         222 ms          222 ms            3
Benchmark_Parallel_Tdigest_Quantile/1024/128/8           0.057 ms        0.057 ms        12200
Benchmark_Parallel_Tdigest_Quantile/4096/128/8           0.197 ms        0.197 ms         3562
Benchmark_Parallel_Tdigest_Quantile/32768/128/8          0.294 ms        0.293 ms         2362
Benchmark_Parallel_Tdigest_Quantile/262144/128/8          2.11 ms         2.11 ms          334
Benchmark_Parallel_Tdigest_Quantile/2097152/128/8         14.1 ms         14.0 ms           50
Benchmark_Parallel_Tdigest_Quantile/16777216/128/8         110 ms          110 ms            6
Benchmark_Parallel_Tdigest_Quantile/1024/256/8           0.062 ms        0.062 ms        11090
Benchmark_Parallel_Tdigest_Quantile/4096/256/8           0.213 ms        0.213 ms         3282
Benchmark_Parallel_Tdigest_Quantile/32768/256/8          0.625 ms        0.624 ms          919
Benchmark_Parallel_Tdigest_Quantile/262144/256/8          2.58 ms         2.58 ms          271
Benchmark_Parallel_Tdigest_Quantile/2097152/256/8         15.0 ms         14.9 ms           47
Benchmark_Parallel_Tdigest_Quantile/16777216/256/8         114 ms          114 ms            6
Benchmark_Parallel_Tdigest_Quantile/1024/128/12          0.063 ms        0.063 ms        10715
Benchmark_Parallel_Tdigest_Quantile/4096/128/12          0.216 ms        0.216 ms         3235
Benchmark_Parallel_Tdigest_Quantile/32768/128/12         0.461 ms        0.459 ms         1502
Benchmark_Parallel_Tdigest_Quantile/262144/128/12         1.49 ms         1.49 ms          468
Benchmark_Parallel_Tdigest_Quantile/2097152/128/12        10.4 ms         10.4 ms           68
Benchmark_Parallel_Tdigest_Quantile/16777216/128/12       80.2 ms         78.7 ms            9
Benchmark_Parallel_Tdigest_Quantile/1024/256/12          0.073 ms        0.073 ms         9368
Benchmark_Parallel_Tdigest_Quantile/4096/256/12          0.223 ms        0.223 ms         3137
Benchmark_Parallel_Tdigest_Quantile/32768/256/12         0.618 ms        0.616 ms         1083
Benchmark_Parallel_Tdigest_Quantile/262144/256/12         2.15 ms         2.15 ms          328
Benchmark_Parallel_Tdigest_Quantile/2097152/256/12        10.6 ms         10.6 ms           66
Benchmark_Parallel_Tdigest_Quantile/16777216/256/12       81.0 ms         79.9 ms            9
Benchmark_Nth_Elem_Quantile/1024                         0.001 ms        0.001 ms      1137602
Benchmark_Nth_Elem_Quantile/4096                         0.002 ms        0.002 ms       349339
Benchmark_Nth_Elem_Quantile/32768                        0.016 ms        0.016 ms        46900
Benchmark_Nth_Elem_Quantile/262144                       0.110 ms        0.110 ms         5588
Benchmark_Nth_Elem_Quantile/2097152                       1.06 ms         1.06 ms          570
Benchmark_Nth_Elem_Quantile/16777216                      17.1 ms         17.1 ms           40
```