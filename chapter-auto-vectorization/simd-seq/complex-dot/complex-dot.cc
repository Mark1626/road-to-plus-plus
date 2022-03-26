#include <benchmark/benchmark.h>

#include "base.cc"
#include "soa.cc"
#include "complex2.cc"
#include "complex4.cc"
#include "complex2-simd.cc"
#include "complex4-simd.cc"

const int st = 128;
const int en = 1<<22;

// const int st = 1<<10;
// const int en = 1<<20;

// const int st = 1<<20;
// const int en = 1<<28;

BENCHMARK(BM_dotprod_stdcomplex_to_stdcomplex)->Range(st, en)
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_dotprod_complexfloat_to_float)->Range(st, en)
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_dotprod_complexfloat_to_stdcomplex)->Range(st, en)
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_dotprod_complexfloat_to_stdcomplex_simd)->Range(st, en)
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_dotprod_complex2_to_stdcomplex)->Range(st, en)
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;
BENCHMARK(BM_dotprod_complex2_to_complex2)->Range(st, en)
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_dotprod_complex4_to_stdcomplex)->Range(st, en)
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_dotprod_simd_stdcomplex_to_stdcomplex)->Range(st, en)
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK(BM_dotprod_simd4_stdcomplex_to_stdcomplex)->Range(st, en)
#ifdef MILLI
->Unit(benchmark::kMillisecond)
#endif
;

BENCHMARK_MAIN();
