project(stsoa LANGUAGES CXX)

cmake_minimum_required(VERSION 3.12)

# add_library(stsoa_complex STATIC
#   simd-complex.hh
#   sse.cc
#   avx.cc
#   )

SET(COMPILE_FLAGS "-march=native -fcx-limited-range -fopenmp")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")

# target_compile_features(stsoa_complex PUBLIC cxx_std_11)

add_executable(stsoa stsoa.cc)

target_link_libraries(stsoa PUBLIC -lcasa_casa)
