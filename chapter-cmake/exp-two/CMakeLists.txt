cmake_minimum_required(VERSION 3.20)

project(exp-two VERSION 1.0)

add_subdirectory(arith)

add_executable(exp-two main.cc)

target_link_libraries(exp-two PUBLIC arith)

target_include_directories(exp-two PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/arith"
                            )
