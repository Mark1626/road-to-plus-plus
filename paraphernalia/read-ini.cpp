#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

using std::cout;
namespace pt = boost::property_tree;

int main() {
  pt::ptree tree;
  pt::read_ini("test.ini", tree);

  auto val = tree.get<std::string>("test");
  cout << "Hello " << val << "\n";
}
