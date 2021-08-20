cmake_minimum_required(VERSION 3.20)

project(benches VERSION 1.0)

set(FLAGS "-O3" "-mssse3" "-mavx2" )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_executable(benchmarks benchmarks.cc)
target_compile_options(benchmarks PRIVATE ${FLAGS})
target_link_libraries(benchmarks PUBLIC benchmark::benchmark -pthread)
target_include_directories(benchmarks PUBLIC
                                  "${PROJECT_BINARY_DIR}")


