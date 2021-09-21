
add_executable(Exp2 main.cc)
target_include_directories(Exp2 PUBLIC /usr/local/include)
target_link_directories(Exp2 PRIVATE /usr/local/lib)
target_link_libraries(Exp2 -lcasa_casa -lcasa_meas -lcasa_measures)
