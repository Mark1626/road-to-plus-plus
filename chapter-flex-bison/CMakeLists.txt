cmake_minimum_required(VERSION 3.20)

project(Lang VERSION 1.0)

find_package(BISON)
find_package(FLEX)

add_subdirectory(exp-three)
add_subdirectory(exp-four)