#pragma once
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <utility>

namespace cxxplot {
class CXXPlot {
  std::ofstream file;
  std::string script_name;
  std::string png_name;

public:
  CXXPlot(std::string script_name, std::string png_name)
      : script_name(script_name), file(script_name), png_name(png_name) {
    if (!file.is_open()) {
      throw std::runtime_error("Error opening file");
    }
    file << "set terminal push\n";
    file << "set terminal png size 400,300 enhanced\n";
    file << "set output '" << png_name << "'\n";
  }
  void script(std::string command) { file << command << "\n"; }
  void export_png() {
    file << "replot\n";
    file << "set terminal pop\n";
    file.flush();

    std::string plot_cmd = "gnuplot -p " + script_name;
    int error = std::system(plot_cmd.c_str());
    if (error) {
      throw std::runtime_error("Error invoking GNU Plot ");
    }
  }
};

template<typename T>
void plot(std::string plotname, std::vector<std::pair<T, T>> data, std::initializer_list<std::string> extras = {}) {
    CXXPlot plot("script.ps", plotname);

    for (auto extra : extras)
      plot.script(extra);


    // Plot the points
    std::ostringstream ss;
    ss << "plot '-' notitle\n";
    for (auto point : data) {
      ss << point.first << " " << point.second << "\n";
    }
    plot.script(ss.str());

    plot.export_png();
}

} // namespace cxxplot
