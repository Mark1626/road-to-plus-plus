
add_executable(Exp4 main.cc)
target_include_directories(Exp4 PUBLIC /usr/local/include)
target_link_directories(Exp4 PRIVATE /usr/local/lib)
target_link_libraries(Exp4 -lcasa_casa -lcasa_meas -lcasa_measures -lcasa_scimath)
