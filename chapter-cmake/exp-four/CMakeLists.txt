cmake_minimum_required(VERSION 3.20)

project(ExpFour VERSION 1.0)

configure_file(Config.h.in Config.h)

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

add_executable(ExpFour main.cc)

add_executable(bench bench.cc)
target_link_libraries(bench PUBLIC benchmark::benchmark -pthread)
target_include_directories(bench PUBLIC
                                  "${PROJECT_BINARY_DIR}")
