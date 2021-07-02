# Chapter 1 - Basics

[Back](../README.md)

## Makefile 101

Set `-Wall -Wextra` to catch a lot of useful compile time warnings

```make
CXXFLAGS=-Wall -Wextra
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

## References

- [CPP Tutor](https://github.com/banach-space/cpp-tutor)
- [auto vs decltype](http://thbecker.net/articles/auto_and_decltype/section_01.html)
- [gslice](http://www.cplusplus.com/reference/valarray/gslice/)
