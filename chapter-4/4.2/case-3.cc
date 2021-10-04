/* 
 * Gist originally by skeeto
 * Modified to distribute workload to GPU
 * https://gist.github.com/skeeto/20d0768222af9e7fe6ec0a2d78726d1a
 * XXTEA block middle collider
 * Usage: $ cc -s -O3 -march=native -fopenmp collide.c && ./a.out
 */
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

static void
xxtea128_encrypt(const uint32_t k[4], uint32_t v[4])
{
    static const uint32_t t[] = {
        0x9e3779b9, 0x3c6ef372, 0xdaa66d2b, 0x78dde6e4, 0x1715609d,
        0xb54cda56, 0x5384540f, 0xf1bbcdc8, 0x8ff34781, 0x2e2ac13a,
        0xcc623af3, 0x6a99b4ac, 0x08d12e65, 0xa708a81e, 0x454021d7,
        0xe3779b90, 0x81af1549, 0x1fe68f02, 0xbe1e08bb,
    };
    for (int i = 0; i < 19; i++) {
        uint32_t e = t[i]>>2 & 3;
        v[0] += ((v[3]>>5 ^ v[1]<<2) + (v[1]>>3 ^ v[3]<<4)) ^
                ((t[i] ^ v[1]) + (k[0^e] ^ v[3]));
        v[1] += ((v[0]>>5 ^ v[2]<<2) + (v[2]>>3 ^ v[0]<<4)) ^
                ((t[i] ^ v[2]) + (k[1^e] ^ v[0]));
        v[2] += ((v[1]>>5 ^ v[3]<<2) + (v[3]>>3 ^ v[1]<<4)) ^
                ((t[i] ^ v[3]) + (k[2^e] ^ v[1]));
        v[3] += ((v[2]>>5 ^ v[0]<<2) + (v[0]>>3 ^ v[2]<<4)) ^
                ((t[i] ^ v[0]) + (k[3^e] ^ v[2]));
    }
}

static uint32_t
hash32(uint32_t x)
{
    x ^= x >> 15; x *= 0xd168aaad;
    x ^= x >> 15; x *= 0xaf723597;
    x ^= x >> 15;
    return x;
}

int
main(void)
{
    printf("Devices: %d\n", omp_get_num_devices());
    int found = 0;
    long long n = 1LL<<30;
    uint32_t seed = hash32(time(0));
    uint32_t k[4] = {
        hash32(seed ^ 1), hash32(seed ^ 2),
        hash32(seed ^ 3), hash32(seed ^ 4),
    };

    printf("seed  = %08lx\n", (long)seed);
    printf("key   = %08lx %08lx %08lx %08lx\n",
           (long)k[0], (long)k[1], (long)k[2], (long)k[3]);

    #pragma omp target data map(to:k) device(0)
    {
        #pragma omp target
        #pragma omp teams
        #pragma omp distribute parallel for
        for (long long i = 0; i < n; i++) {
            uint32_t x = hash32(seed ^ (uint32_t)(i>>30));
            uint32_t b[4] = {
                x ^ hash32(i*4 + 0), x ^ hash32(i*4 + 1),
                x ^ hash32(i*4 + 2), x ^ hash32(i*4 + 3),
            };
            xxtea128_encrypt(k, b);
            if (b[1] == b[2]) {
                #pragma omp critical
                {
                    found = 1;
                    printf("i     = %lld\n", i);
                    printf("value   = %08lx %08lx %08lx %08lx\n",
                        (long)b[0], (long)b[1], (long)b[2], (long)b[3]);
                }
                #pragma omp cancel for
            }
            if (found) {
                #pragma omp cancel for
            }
        }
    }

}