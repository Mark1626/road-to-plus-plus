cmake_minimum_required(VERSION 3.20)

project(ExpThree VERSION 1.0)

configure_file(Config.h.in Config.h)

add_executable(ExpThree main.cc)

target_include_directories(ExpThree PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            )
