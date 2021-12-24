
add_executable(Exp6 main.cc)
target_include_directories(Exp6 PUBLIC "${CMAKE_CURRENT_SOURCES_DIR}")
target_link_libraries(Exp6 -lcasa_casa -lcasa_fits -lcasa_images -lcasa_coordinates -lcasa_lattices -lcfitsio -lpthread)
