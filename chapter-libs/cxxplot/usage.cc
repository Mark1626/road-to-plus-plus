#include "cxxplot.hh"

int main() {
  CXXPlot plot("script.ps", "test.png");
  plot.script("set key right nobox\n");
  plot.script("set samples 100\n");
  plot.script(
      "plot [-pi/2:pi] cos(x),-(sin(x) > sin(x+1) ? sin(x) : sin(x+1))\n");
  plot.export_png();

  return 0;
}
