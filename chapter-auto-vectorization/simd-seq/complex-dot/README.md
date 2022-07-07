```
2022-03-25T13:29:11+05:30
Running ./complex-dot/complex-dot
Run on (12 X 2600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 256 KiB (x6)
  L3 Unified 12288 KiB (x1)
Load Average: 1.45, 1.58, 1.91
---------------------------------------------------------------------------------------------
Benchmark                                                   Time             CPU   Iterations
---------------------------------------------------------------------------------------------
BM_dotprod_stdcomplex_to_stdcomplex/1024                 1745 ns         1742 ns       394182
BM_dotprod_stdcomplex_to_stdcomplex/4096                 6933 ns         6926 ns       100059
BM_dotprod_stdcomplex_to_stdcomplex/32768               56072 ns        56024 ns        12130
BM_dotprod_stdcomplex_to_stdcomplex/262144             611467 ns       607113 ns         1000
BM_dotprod_stdcomplex_to_stdcomplex/1048576           3213720 ns      3193363 ns          281
BM_dotprod_complexfloat_to_float/1024                    1699 ns         1697 ns       355330
BM_dotprod_complexfloat_to_float/4096                    6600 ns         6596 ns       102375
BM_dotprod_complexfloat_to_float/32768                  58069 ns        58036 ns        11031
BM_dotprod_complexfloat_to_float/262144                592170 ns       590538 ns         1254
BM_dotprod_complexfloat_to_float/1048576              3175336 ns      3164324 ns          296
BM_dotprod_complexfloat_to_stdcomplex/1024                203 ns          202 ns      3411971
BM_dotprod_complexfloat_to_stdcomplex/4096               1193 ns         1191 ns       582843
BM_dotprod_complexfloat_to_stdcomplex/32768             14466 ns        14458 ns        48254
BM_dotprod_complexfloat_to_stdcomplex/262144           117474 ns       117417 ns         5598
BM_dotprod_complexfloat_to_stdcomplex/1048576          936533 ns       936167 ns          711
BM_dotprod_complexfloat_to_stdcomplex_simd/1024           221 ns          221 ns      3072264
BM_dotprod_complexfloat_to_stdcomplex_simd/4096          1176 ns         1176 ns       580042
BM_dotprod_complexfloat_to_stdcomplex_simd/32768        14736 ns        14723 ns        49757
BM_dotprod_complexfloat_to_stdcomplex_simd/262144      120090 ns       120004 ns         5297
BM_dotprod_complexfloat_to_stdcomplex_simd/1048576     974755 ns       974548 ns          650
BM_dotprod_complex2_to_stdcomplex/1024                   1278 ns         1276 ns       526779
BM_dotprod_complex2_to_stdcomplex/4096                   5071 ns         5065 ns       136591
BM_dotprod_complex2_to_stdcomplex/32768                 40952 ns        40923 ns        16979
BM_dotprod_complex2_to_stdcomplex/262144               332749 ns       332502 ns         2069
BM_dotprod_complex2_to_stdcomplex/1048576             1730890 ns      1730330 ns          409
BM_dotprod_complex2_to_complex2/1024                     1258 ns         1256 ns       538536
BM_dotprod_complex2_to_complex2/4096                     5237 ns         5232 ns       135179
BM_dotprod_complex2_to_complex2/32768                   41919 ns        41907 ns        16991
BM_dotprod_complex2_to_complex2/262144                 336011 ns       335881 ns         1965
BM_dotprod_complex2_to_complex2/1048576               1765217 ns      1764603 ns          393
BM_dotprod_complex4_to_stdcomplex/1024                   1361 ns         1360 ns       501961
BM_dotprod_complex4_to_stdcomplex/4096                   5459 ns         5455 ns       123333
BM_dotprod_complex4_to_stdcomplex/32768                 44965 ns        44924 ns        15756
BM_dotprod_complex4_to_stdcomplex/262144               360060 ns       359768 ns         1915
BM_dotprod_complex4_to_stdcomplex/1048576             2026095 ns      2024170 ns          341
```


