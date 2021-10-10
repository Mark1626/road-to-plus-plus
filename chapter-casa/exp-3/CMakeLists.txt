
add_executable(Exp3 main.cc)
target_include_directories(Exp3 PUBLIC /usr/local/include)
target_link_directories(Exp3 PRIVATE /usr/local/lib)
target_link_libraries(Exp3 -lcasa_casa -lcasa_meas -lcasa_measures)
