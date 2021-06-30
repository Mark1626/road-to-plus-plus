#include <cstdint>

bool is_prime(std::int32_t n) {
  if(!(n & 1))
    return false;

  for (std::int32_t p = 3; p * p <= n; p += 2)
    if (n % p == 0)
      return false;

  return true;
}
