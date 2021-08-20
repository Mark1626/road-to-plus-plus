cmake_minimum_required(VERSION 3.20)

project(fractal VERSION 1.0)

option(ENABLE_OPENMP ON)
option(ANIMATE OFF)
option(DEBUG OFF)

# set(FLAGS "-Wall" "-Wextra" "-O3" "-ftree-vectorize" )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

