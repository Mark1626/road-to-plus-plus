cmake_minimum_required(VERSION 3.20)

project(exp-three VERSION 1.0)

option(USE_SSE "Enable SSE in computation" ON)

configure_file(Config.h.in Config.h)

if(USE_SSE)
  add_subdirectory(sandpile_sse)
  list(APPEND EXTRA_LIBS sandpile_sse)
  list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/sandpile_sse")
endif()

add_subdirectory(sandpile)

target_link_libraries(exp-three PUBLIC ${EXTRA_LIBS})

add_executable(exp-three exp-sandpile.cc)

