cmake_minimum_required(VERSION 3.20)

project(ExpSeven VERSION 1.0)

find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)

set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(hello_py MODULE hello.cc)

target_link_directories(hello_py PRIVATE /usr/local/lib)
target_link_libraries(hello_py ${Boost_Libraries} ${PYTHON_LIBRARIES} -lboost_python39)
target_include_directories(hello_py
  PRIVATE ${PYTHON_INCLUDE_DIRS}
  PRIVATE /usr/local/include
)
