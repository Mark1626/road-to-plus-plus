# Learning CMake

- **Exp-1:** A basic build of the target target
- **Exp-2:** Building a target with library dependency
- **Exp-3:** External config
- **Exp-4:** CMake setup with Google Benchmark and Google Test available
- **Exp-5:** Setting C-compiler path and different flags for debug compilation and release compilation
  + In the example the `CMAKE_CXX_FLAGS_DEBUG` is set to include `-g` so that debug symbols are exposed
  + `cmake -DCMAKE_C_COMPILER:FILEPATH=/path/to/gcc-11 -DCMAKE_CXX_COMPILER:FILEPATH=/path/to/g++-11 --config Debug ..`
  + `export CC=/path/to/gcc-11`
  + `export CXX=/path/to/gcc-11`
  + `make <target>`

> **Note:** `CMAKE_C_COMPILER` **should not be set** in `CMakeLists` with `set` instead we pass it as an argument and get it checked in `CMakeCache`

See [this question](https://discourse.cmake.org/t/proper-way-to-set-compiler-and-language-standard-in-cmake/2756) and [this stackoverflow question](https://stackoverflow.com/questions/17275348/how-to-specify-new-gcc-path-for-cmake)
- **Exp-6:** Use a library installed in `/usr/local/lib`
- **Exp-7:** Boost python CMake
- **Exp-8:** Run a command from within `CMakeLists.txt`
- **Exp-9:** Cache variable

## Reading

### Tutorials

- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/A%20Basic%20Starting%20Point.html)
- [Using CMake](https://eliasdaler.github.io/using-cmake/)
