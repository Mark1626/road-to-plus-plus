link_libraries(ppm)
include_directories(ppm_random_distribution PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/ppm"
                            )


add_executable(ppm_random_distribution ppm_random_distribution.cc)
add_executable(ppm_write_text ppm_write_text.cc)
