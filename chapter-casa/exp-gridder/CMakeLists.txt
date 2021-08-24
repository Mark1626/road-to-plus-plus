
add_executable(gridder gridder.cc)
target_include_directories(gridder PUBLIC "${CMAKE_CURRENT_SOURCES_DIR}")
target_link_libraries(gridder -lcasa_ms -lcasa_measures -lcasa_tables -lcasa_casa)
