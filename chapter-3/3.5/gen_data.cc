#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Value = float;

int main() {

  const int N = 1 << 24;
  const int num_sections = 8;
  const int section_width = N / num_sections;

  std::fstream ofs("data.dat", std::ios::out);

  {
    std::uniform_real_distribution<> reals(0.0, 1.0);
    std::random_device gen;

    std::vector<Value> values;

    for (int i = 0; i < N; i++) {
      values.push_back(reals(gen));
    }

    boost::archive::text_oarchive oa(ofs);

    for (int i = 0; i < N; i++) {
      oa << values[i];
    }

    // Too lazy to write a deserialize function that can read ranges, so this will have to do
    for (int section = 0; section < num_sections; section++) {
      std::string filename = "data_" + std::to_string(section) + ".dat";
      std::fstream local_ofs(filename, std::ios::out);
      boost::archive::text_oarchive local_oa(local_ofs);

      for (int i = 0; i < section_width; i++) {
        int idx = (section * section_width) + i;
        local_oa << values[idx];
      }
    }
  }

  std::vector<Value> restore_values;
  {
    std::ifstream ifs("data.dat");
    boost::archive::text_iarchive ia(ifs);

    for (int i = 0; i < N; i++) {
      Value value;
      ia >> value;
      restore_values.push_back(value);
    }
  }

  for (int i = 0; i < 10; i++) {
    std::cout << restore_values[i] << " ";
  }
  std::cout << std::endl;
}