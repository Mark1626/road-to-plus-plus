cmake_minimum_required(VERSION 3.20)

project(ExpSix VERSION 1.0)

find_package(fmt REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(main main.cc)

target_link_directories(main PUBLIC /usr/local/lib)
target_link_libraries(main PUBLIC fmt)
target_include_directories(main PRIVATE /usr/local/include)
