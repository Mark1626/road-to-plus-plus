#include <boost/python.hpp>

char const* hello() {
  return "Hello World";
}

BOOST_PYTHON_MODULE(hello_py) {
  boost::python::def("hello", hello);
}