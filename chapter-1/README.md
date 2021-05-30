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

## References

- [CPP Tutor](https://github.com/banach-space/cpp-tutor)
- [auto vs decltype](http://thbecker.net/articles/auto_and_decltype/section_01.html)
