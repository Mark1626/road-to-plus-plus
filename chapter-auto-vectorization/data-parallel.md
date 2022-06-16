
# On data parallelism

Inline ASM > Intrinsics > Explicit auto vectorization through hints -> Implicit auto vectorization

For a given problem there are multiple ways to add data parallelism.


**Inline ASM**

The most efficient data parallel solution would be from inline ASM, we can manager our own register, and utilize instuction otherwise not available through instrincs. This is not scalable, but there are some rare cases we need this.

**Intrinsics**

Intrinsics are high wrappers are compiled done into ASM, management of registers is done by the compiler. Intrinsics are the best option for high performance dependant code. The major drawback would include complexity of the code, portability, and the time need to assert memory safety of code.

**Explicit auto vectorization**

We can ask the compiler to perform explicit auto vectorization through hints eg) `__restrict__` to mention that there will be no overlaps(this reduces the ASM generated), `alignas` to align data structures.

OpenMP has the directive `#pragma omp simd` which can be used to make the compiler consider vectorization. This is useful in cases where the compiler cannot auto-vectorize without information of the workload, such as the **SIMD N Distinct Elements** problem

[SIMD N Distinct Elements](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM1ykrgAyeAyYAHI%2BAEaYxCAAzKQADqgKhE4MHt6%2B/oGp6Y4CoeFRLLHxSXaYDplCBEzEBNk%2BfgG2mPZFDPWNBCWRMXGJtg1NLbntCmP9YYPlwwkAlLaoXsTI7BxoDNMA1GEEewp7JgkAIntcGmdWGgCCAG6oeOh7VF4MyBCHewBUAH0AcRMNNiHgHEC9kxSAdBP8gSCwRCCFDorDDoDgaCCODIQC9ugMfCIktTgB2O73PY005mBLJYhMYAsJh7VAsZLHPAsdAmB6094kPY/eF4U4XPY3BIWA4Stx7CK3A7WaxkkyU/nUwW037aCWXPkAVgseBMRvOty1OtpTHNFm05suqslJz%2Be2i9rNFqtAtpGstDwDWo4K1onCNvD8HC0pFQnDcqssxzWG0wdISPFIBE0oZWCEwTCw8QgKwA1hIAJwAOiN5MrXC4RqblYCmcr%2Bk4kl4LAkGg0pGjsfjHF4ChAA5zMdDpDgsCQaE5dDi5Eoi%2BSy/iyAMRi4Zn7fDoBDi44g0VzpGiYUaAE9OFnF2xBAB5Bi0O/T0hYVlGcSf/AgrUjyghemCqDUXjHvevCHJ00H6Hg0RMsQN4eFg8G4jyuYrFQBjAAoABqeCYAA7s%2BySMPB/CCCIYjsFIMiCIoKjqJ%2BuhmPohjGEmNi0Ih46wMwbAgEyRigo8STAfEuKfGWSwrKgyTdOOHAALTPgkeyqVQDCoKpwEOCQeAAF7ptpimMJyY6dDU3QuAw7ieK0eghHMZQVHoBQZAIEx%2BPkaTeQwAzucMgTVLUAi9OMTm5GFNkRT0MzBUM8RhTMvl6NMfTJQsqUrAoqabHouKYFsPBhhGUYXiOqgABwAGyqfVkh7NuXFXGY1YaF1IqJpY1iwrghDCuYmawh4S70MQGZmEsvBTlo8mkBWZi1bWtW1VwlYJAkki7btjbkp2HDdoO1WcGOE7Zths4wIgKAchuU2rhA66biAwCNoENC0MexCnuen5XswKHwY%2BjAEK%2B74Xt%2BXF/rGAG2XgwEqbGYEQVB3AwYIcGfnxSG3mhWyxphvZYzheGEcRZEUdGWbUcIojiAxDPMWoF66EkO7cf1lgIdEAmlnGSmZCp6maeZtNWR0XSZPZjk5H4RpBA5OUeVw9UpAF3QZcrXndGroWa%2BF3RRc0MVKzLSORUlbkpRIxvpRbIDK1lTSG6l9X5YV9HZiCZUzuGHCRmdn41Q1TUtcAyDIFcXDVlwvU8YN%2BBENNo2BHsE1PXEGbLPN2Hlv45IJ5I9XkuSXC1QkGjtvWx2nUOvAjpdk43XO91vc9FCvY971tbu%2B4Dj9f0AxewO3mDHJPpDb4fgjmA/sA8O8IjQEgZ%2B6PIJBAfkDjQexvjyGoRgxPzeCZPlXwlNEaR5GUVjjGM3REjSKzSjs2xejcygyf44LCkRYCDFhpaystnAQFcBlQIrlSj238oUTI0CtaIOKHbXKmV4qmydorTB4DErZXQerUYfRkFu1mHAjBXBvbrCKoEEqAcKrByqmHTgdVGrNVatzDqXUeoQD6lYPmewhppwzJnbOm4xFzWutOJaK1JDVnqrVI0DZKz1XLrXMwDYuZdhYcOC6tgroLTzMdMwejm4GOMUtKS6RnCSCAA%3D)

**Implicit auto vectorization**

Modern complier can auto-vectorize most common scenarios. In cases where there is no guarentee of 
pointer overlap it generates two versions of the function one vectorized, another not vectorized and utilizes the necessary version after checking for overlaps.

