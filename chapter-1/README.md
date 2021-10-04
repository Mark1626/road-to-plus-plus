# Chapter 1 - Basics

[Back](../README.md)

## Compilation 101

### Essential Flags

#### -Wall -Wextra -Werror

Set `-Wall -Wextra` to catch a lot of useful compile time warnings

```make
CXXFLAGS=-Wall -Wextra
```

`-Werror` makes the compiler treat warnings as errors

#### -std= -pedantic

The flag `-std` defines the C / C++ standard

```sh
c++ -o test test.cc -std=c++14 -pedantic

cc -o test test.c -std=c99 -pedantic
```

The `-pedantic` will cause the compiler to throw an error when using features which conflict with ISO C or ISO C++

#### -O<n>

Defines the level of optimization

```sh
-Og
-O0
-O1
-O2
-O3
-Ofast
```

`-O3` should be easy enough to use for most cases
`-Ofast` is `-O3` + `-ffast-math`, fast math adds some float and double optimization that are unsafe use this only when you know what you are doing


#### -g -fno-omit-frame-pointer

`-g` turns on debugging symbols, you will need this to connect your application to a debugger

`-fno-omit-frame-pointer` Enables frame pointer, this is a very essential flag for profiling applications

### Reading flags from a file

```
c++ $(< flags) test.cc -o test
```


## Preprocessor 101

### Include Guard

```cpp
// header.hh
#ifdef HEADER_HH
#define HEADER_HH

// Method to expose

#endif
```

```cpp
// header.hh
#pragma once

// Method to expose

```


## Basics 101

Brushing up some features

### Initializer list

Short hand to initialize members of a class

```cpp
class Test {
  int n;
  char c;
  public:
    Test() : n(1), c{'o'} {}
    Test(n) : n(n) {}
    Test(Test& a) : n(a.n) {}
}
```

### Move ctor

Move ctor steals ownership of dynamic resources of source to the target class

```cpp
class A {
  string s;
  int n;
  public:
  A(A&& o) : s(std::move(o.s)), n(std::exchange(o.n, 0)) {}
}
```

### Trailing return type

```cpp
auto add(int a, int b) -> int {
  return a + b;
}
```

### Remove reference

```cpp
  std::cout << std::is_same(int, std::remove_reference<int>::type)
  std::cout << std::is_same(int, std::remove_reference<int &>::type)
  std::cout << std::is_same(int, std::remove_reference<int &&>::type)
```

### auto vs decltype

#### auto

A simple usecase of auto

```cpp
std::vector<std::string> x;

for (std::vector<std::string>::iterator it = x.begin(), it != x.end(); ++it) { }
// or
for (auto it = x.begin(), it != x.end(); ++it) { }
```

#### decltype

```cpp
template <typename A, typename B>
auto add(A a, B a) -> decltype(a * b) {
  return a + b
}
```

```cpp
template <typename A, typename B>
auto min(A a, B b) -> typename std::remove_reference<decltype(x < y ? x : y)>::type {
  return x < y ? x : y;
}
```

#### Comparison between auto and decltype

```cpp
int x;
const int& crx = x;

typedef decltype(x) type_x;
// Type is int
type_x a1 = 10;
// Type is int
auto a2 = x;

typedef decltype(crx) type_crx;
// Type is const int&
type_crx b1 = 10;
// Type is int
auto b2 = crx
```

### valarray - slice and gslice

#### slice

```cpp

  0  1  2  3
  4  5  6  7
  8  9 10 11
  12 13 14 15

  std::valarray<int> val(16);

  // Row 0, 1, 2, 3
  val[std::slice(0, 4, 1)];

  // Column 0, 4, 8, 12
  val[std::slice(0, 4, 4)];

  // Major Diagonal 0, 5, 10, 15
  val[std::slice(0, 4, 5)];

  // Minor Diagonal 3, 6, 9, 12
  val[std::slice(3, 4, 3)];
```

#### gslice

```
Assume a 2 D array

Example

 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15

-------------------

val[std::gslice(0, {4, 2}, {4, 1})]

                    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]
start=0:             *
                     |
size=4, stride=4:    *-----------*-----------*-------------*------------*
                     |           |           |             |            |
size=2, stride=1:    *--*        *--*        *--*          *---*        *---*
                     |  |        |  |        |  |          |   |        |   |
gslice:              *  *        *  *        *  *          *   *        *   *
                    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]

-------------------

Rows

val[std::gslice(0, {4}, {1})]

                    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]
start=0:             *
                     |
size=4, stride=1:    *-----------*-----------*-------------*------------*
                     |           |           |             |            |
gslice:              *           *           *             *            *
                    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]

-------------------

Column

val[std::gslice(0, {4}, {4})]

                    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]
start=0:             *
                     |
size=4, stride=4:    *-----------*-----------*-------------*------------*
                     |           |           |             |            |
gslice:              *           *           *             *            *
                    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]

-------------------

Major Diagonal

  0  1  2  3
  4  5  6  7
  8  9 10 11
  12 13 14 15

  0, 5, 10, 15

val[std::gslice(0, {4}, {5})]

                    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]
start=0:             *
                     |
size=4, stride=5:    *--------------*--------------*--------------------*
                     |              |              |                    |
gslice:              *              *              *                    *
                    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]

-------------------

Minor Diagonal

  0  1  2  3
  4  5  6  7
  8  9 10 11
  12 13 14 15

 3, 6, 9, 12

val[std::gslice(3, {4}, {3})]

                    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]
start=3:                      *
                              |
size=4, stride=3:             *--------*--------*-----------*
                              |        |        |           |
gslice:                       *        *        *           *
                    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]
```

### Rule of Zero

Classes that have custom destructors, copy/move assignment should deal with ownership exclusively. Other classes should not have touch with the ownership

```cpp
class Test {
  std::vector arr;
  public:
  Test(std::vector arr) : arr(arr) {}; // We rely on std::vector to manage it's own resources
};
```

### Rule of Five

When creating a class which is going to manage it's own resources 
always define these 5 member functions
2. Destructor
3. Copy Ctor
3. Copy Assignment
4. Move Ctor
5. Move Assignment

### Packing and Data Structure Alignment

```cpp
struct SAO_Body {
  int xno;
  double sra0;
  double sdec0;
  char is[2];
  unsigned short mag;
  float xrpm;
  float xdpm;
};

assert(sizeof(SAO_Body) == 32) // This is not guaranteed, and differs machine to machine, it was 40 for me

struct __attribute__((__packed__)) SAO_Body {
  int xno;
  double sra0;
  double sdec0;
  char is[2];
  unsigned short mag;
  float xrpm;
  float xdpm;
};

assert(sizeof(SAO_Body) == 32) // This struct is packed and guaranteed to be 32 bytes,
```

> Note: In case your wondering the above struct was used to parse a binary data inside the SAO_Catalog

### Endianess

```cpp
#if defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define htobe32(x) OSSwapHostToBigInt32(x)

#elif defined(__linux__)
#include <endian.h>
#endif

uint8_t* SHA1::digest() {
  padBlock();

  #ifdef LITTLE_ENDIAN
    state[0] = htobe32(state[0]);
    state[1] = htobe32(state[1]);
    state[2] = htobe32(state[2]);
    state[3] = htobe32(state[3]);
    state[4] = htobe32(state[4]);
  #endif

  return (uint8_t*)state;
}
```

## References

- [CPP Tutor](https://github.com/banach-space/cpp-tutor)
- [auto vs decltype](http://thbecker.net/articles/auto_and_decltype/section_01.html)
- [gslice](http://www.cplusplus.com/reference/valarray/gslice/)
