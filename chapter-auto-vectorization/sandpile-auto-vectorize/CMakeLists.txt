cmake_minimum_required(VERSION 3.20)

project(fractal VERSION 1.0)

configure_file(Config.h.in Config.h)

option(ENABLE_OPENMP ON)
option(ANIMATE OFF)

enable_testing()

include(FetchContent)

FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.10.x)

FetchContent_Declare(googlebenchmark
                     GIT_REPOSITORY https://github.com/google/benchmark
        )
FetchContent_MakeAvailable(googletest googlebenchmark)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_subdirectory(sandpile)

add_executable(fractal main.cc)
target_link_libraries(fractal PUBLIC sandpile)
target_include_directories(fractal PUBLIC
                                  "${PROJECT_BINARY_DIR}"
                                  "${PROJECT_SOURCE_DIR}/sandpile")


add_executable(bench bench.cc)
target_link_libraries(bench PUBLIC sandpile benchmark::benchmark -pthread)
target_include_directories(bench PUBLIC
                                  "${PROJECT_BINARY_DIR}"
                                  "${PROJECT_SOURCE_DIR}/sandpile")
