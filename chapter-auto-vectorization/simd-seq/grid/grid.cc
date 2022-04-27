#include <benchmark/benchmark.h>
#include "base.cc"
#include "ptr.cc"
#include "complex2-simd.cc"
#include "complex4-simd.cc"
#include "complex-simd-v2.cc"
#include "complex-simd-v3.cc"

// const int supportSt = 8;
const int support = 128;

const int st = 2 * support;
const int en = 4 * support;

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

BENCHMARK(BM_grid_simd_v2)->Ranges({{st, en}, {support, support}})
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_grid_simd_v2_4)->Ranges({{st, en}, {support, support}})
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_grid_simd_v3)->Ranges({{st, en}, {support, support}})
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_grid_simd_v3_4)->Ranges({{st, en}, {support, support}})
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK_MAIN();
