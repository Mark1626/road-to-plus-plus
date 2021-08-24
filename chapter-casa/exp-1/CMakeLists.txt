
add_executable(Exp1 main.cc)
target_include_directories(Exp1 PUBLIC /usr/local/include)
target_link_directories(Exp1 PRIVATE /usr/local/lib)
target_link_libraries(Exp1 -lcasa_casa -lcasa_meas -lcasa_measures)
