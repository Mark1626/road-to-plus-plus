#include <benchmark/benchmark.h>
#include "base.cc"
#include "ptr.cc"
#include "complex2-simd.cc"
#include "complex4-simd.cc"

// const int supportSt = 8;
const int support = 128;

const int st = 2 * support;
const int en = 8 * support;

BENCHMARK(BM_grid_std_complex)->Ranges({{st, en}, {support, support}})
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_grid_ptr)->Ranges({{st, en}, {support, support}})
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_grid_simd)->Ranges({{st, en}, {support, support}})
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_grid_simd_4)->Ranges({{st, en}, {support, support}})
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK_MAIN();
