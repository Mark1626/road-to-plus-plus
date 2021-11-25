#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

typedef uint8_t u8;

class PPM {
public:
  int width;
  int height;
  int depth;
  PPM(int width, int height, int depth)
      : width(width), height(height), depth(depth) {}
};

namespace boost {
namespace serialization {
template <class Archive>
void serialize(Archive &ar, PPM &g, const unsigned int version) {
  ar & "P6\n";
  ar & g.width & " " & g.height & "\n";
  ar & g.depth & "\n";

}
} // namespace serialization
} // namespace boost

int main() {
  PPM ppm(8, 8, 256);
  std::fstream file("output.txt", std::ios::out);
  boost::archive::text_oarchive oa(file);

  oa << ppm;
}
