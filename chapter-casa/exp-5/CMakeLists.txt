
add_executable(Exp5 main.cc)
target_include_directories(Exp5 PUBLIC "${CMAKE_CURRENT_SOURCES_DIR}")
target_link_libraries(Exp5 -lcasa_casa -lcasa_fits -lcasa_images -lcasa_coordinates -lcasa_lattices)
