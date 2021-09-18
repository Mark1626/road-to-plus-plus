# Benchmarks

[Back](../index.md)

## Benchmark abs implementations

[Godbolt](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAM1QDsCBlZAQwBtMQBGAFlJvoCqAZ0wAFAB4gA5AAYppAFZdSrZrVDIApACYAQjt2kR7ZATx1KmWugDCqVgFcAtrS4BmUlfQAZPLUwAcs4ARpjEIADspAAOqEKE5rR2ji7uMXEJdL7%2BQU6h4VHGmKaJDATMxATJzq6cHkUldGUVBNmBIWGRRuWV1al13S1tufmRAJRGqA7EyBxSOm5%2ByI5YANSabjbITswECBvYmjIAgkfHfgSrzMFCnBAXV2PrEfonq%2B8PQutuACJX34c3NhVm5OBtXsd3v9NABWQF/ITgs5Q5irAC0GwRSLe72ImAI01oV2xpwiPzOZwe1yE2nu9EezwhUNWeIJxCJ1IgzDGJM0ZKkE1Y0hh8lcsnkqGkNgMBlWQimM0w620oPkBGkcjGEwA1iBtBEAHQyAAcxphxoAnGaeJwYREAGwqaTceROEAWi0Gj3en2%2B0hiuSkSVSeRCEAyUjq8UTOCwJBoJzRPDsMgUCAJpMplCqdScbQyTh8ZMEMJhiDBDXyYJ%2BCoAT2knHkCacVgIAHlaKx6%2BLSFgdup2JXe3g8SUAG6YMM9zDiYoOEsN%2BQXTBCnusPDBYh1uxYIcEYh4N1SRsTKiqYBCABqeEwAHc29ErIu%2BHRGCxBzwX4IRBIh0pCzmGgynoKgbmGkATKg0RmHQU5om2bjolQ%2B6YJgaITqYJB4AAXpgoYrsUMGuBAXh9LU2ieNYwwdOEnARrE8REWRXAUQxmS0NReSdHRRgEY0tDNL09g1CxvEmERgmtH47RcbREZCD0VTCf0FEKUM0kjNxMgTPK0yzFwgrCqKQ7BuIxr2mi9rcKsyxqMAqx5kaBqcKsEDSnoBikKsuCECQyqgl5diJsmYT%2BZwYxqpWWqkLqbienaxoyA6xq2toMIyBa2i8KuLqkG6dERgGErSKG4aRlFpCxogKCoMFKbkJQGYheEwCcHmRasCWxBlhWPbVrQdbPs2rYdl2Q59nZg49vgo5mBOU6BjOc4LseS70CuQ7rpu24YHMgb7oei6nueV43vej5io2X5vmwHCfvwBDCGIkg9koFGASgwGGFt4EQJB0GJHBCH4eJiSWNYzGFl4nGjIWbFMcptTpIxiQw9xYmEaUimQxj/GSWjtGDEJKRI2plQEwZkx6XdhlSCK/omdIZkWVZqzAMgyAOZwBraK57n6CB3n4EQxBhYFtWZqFCzaBF5XRjqIBuG4BpK6ravq06Ui5UVQYlUYZVRpqtPaMZPbBpF8ukBO3Vg9wQA%3D%3D%3D)

```
N = 1 << 16

------------------------------------------------------------
Benchmark                  Time             CPU   Iterations
------------------------------------------------------------
BM_abs_Bitwise          9533 ns         9532 ns        72129
BM_abs_Auto_Vec         9522 ns         9520 ns        74501
BM_abs_Manual_SSE       4336 ns         4336 ns       177866
BM_abs_Manual_AVX       1281 ns         1280 ns       452267
```

```
------------------------------------------------------------------
Benchmark                        Time             CPU   Iterations
------------------------------------------------------------------
BM_abs_Bitwise/256            17.7 ns         17.7 ns     39689740
BM_abs_Bitwise/1024           71.8 ns         71.8 ns      9724924
BM_abs_Bitwise/4096            264 ns          264 ns      2639866
BM_abs_Bitwise/16384          1289 ns         1288 ns       540336
BM_abs_Bitwise/65536          9548 ns         9547 ns        74612
BM_abs_Bitwise_BigO        2238.01 (1)     2237.74 (1)
BM_abs_Bitwise_RMS             165 %           165 %
BM_abs_Auto_Vec/256           11.4 ns         11.4 ns     60547872
BM_abs_Auto_Vec/1024          50.6 ns         50.6 ns     13646022
BM_abs_Auto_Vec/4096           180 ns          179 ns      3894969
BM_abs_Auto_Vec/16384         1280 ns         1280 ns       473405
BM_abs_Auto_Vec/65536         9599 ns         9598 ns        72270
BM_abs_Auto_Vec_BigO       2224.05 (1)     2223.89 (1)
BM_abs_Auto_Vec_RMS            167 %           167 %
BM_abs_Manual_SSE/256         8.57 ns         8.56 ns     80760533
BM_abs_Manual_SSE/1024        32.6 ns         32.5 ns     21339121
BM_abs_Manual_SSE/4096         135 ns          135 ns      5216951
BM_abs_Manual_SSE/16384        701 ns          701 ns       929134
BM_abs_Manual_SSE/65536       4228 ns         4227 ns       176159
BM_abs_Manual_SSE_BigO     1020.91 (1)     1020.77 (1)
BM_abs_Manual_SSE_RMS          159 %           159 %
BM_abs_Manual_AVX/256         2.41 ns         2.41 ns    287845517
BM_abs_Manual_AVX/1024        8.51 ns         8.51 ns     83830327
BM_abs_Manual_AVX/4096        32.5 ns         32.5 ns     21076846
BM_abs_Manual_AVX/16384        234 ns          234 ns      2840840
BM_abs_Manual_AVX/65536       1216 ns         1216 ns       532133
BM_abs_Manual_AVX_BigO      298.81 (1)      298.75 (1)
BM_abs_Manual_AVX_RMS          156 %           156 %
```

## Benchmark max

```
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_max_Bitwise/256             19.2 ns         19.2 ns     35336224
BM_max_Bitwise/512             44.4 ns         44.4 ns     15927879
BM_max_Bitwise/4096             479 ns          479 ns      1555960
BM_max_Bitwise/32768           6068 ns         6067 ns       118922
BM_max_Bitwise/262144         52604 ns        52602 ns        13167
BM_max_Bitwise/2097152      1098155 ns      1098032 ns          647
BM_max_Bitwise/16777216    10776105 ns     10774908 ns           65
BM_max_Bitwise_BigO      1704782.06 (1)  1704593.06 (1)
BM_max_Bitwise_RMS              218 %           218 %
BM_max_Std_algo/256            12.5 ns         12.5 ns     52405801
BM_max_Std_algo/512            24.5 ns         24.5 ns     28381332
BM_max_Std_algo/4096            438 ns          438 ns      1606676
BM_max_Std_algo/32768          5950 ns         5948 ns       121275
BM_max_Std_algo/262144        53775 ns        53763 ns        13370
BM_max_Std_algo/2097152     1119276 ns      1119220 ns          649
BM_max_Std_algo/16777216   11485033 ns     11483812 ns           64
BM_max_Std_algo_BigO     1809215.64 (1)  1809031.26 (1)
BM_max_Std_algo_RMS             219 %           219 %
BM_max_Normal/256              17.3 ns         17.3 ns     38736954
BM_max_Normal/512              33.5 ns         33.5 ns     21028662
BM_max_Normal/4096              431 ns          431 ns      1657585
BM_max_Normal/32768            5818 ns         5818 ns       120598
BM_max_Normal/262144          53512 ns        53503 ns        13499
BM_max_Normal/2097152       1130510 ns      1130352 ns          651
BM_max_Normal/16777216     11521741 ns     11507810 ns           63
BM_max_Normal_BigO       1816009.00 (1)  1813994.82 (1)
BM_max_Normal_RMS               219 %           219 %
BM_max_SSE/256                 24.6 ns         24.6 ns     28041277
BM_max_SSE/512                  150 ns          150 ns      4585083
BM_max_SSE/4096                1239 ns         1239 ns       561644
BM_max_SSE/32768              10493 ns        10491 ns        68852
BM_max_SSE/262144             81392 ns        81379 ns         8524
BM_max_SSE/2097152           653613 ns       653598 ns         1058
BM_max_SSE/16777216         5696835 ns      5696182 ns          121
BM_max_SSE_BigO           920535.31 (1)   920437.75 (1)
BM_max_SSE_RMS                  213 %           213 %
BM_max_AVX/256                 13.2 ns         13.2 ns     52096513
BM_max_AVX/512                 26.4 ns         26.4 ns     26651952
BM_max_AVX/4096                 824 ns          824 ns      1000000
BM_max_AVX/32768               6914 ns         6913 ns       102474
BM_max_AVX/262144             55971 ns        55964 ns        12460
BM_max_AVX/2097152           442594 ns       442505 ns         1580
BM_max_AVX/16777216         3812488 ns      3811545 ns          187
BM_max_AVX_BigO           616975.73 (1)   616827.39 (1)
BM_max_AVX_RMS                  213 %           213 %
```
