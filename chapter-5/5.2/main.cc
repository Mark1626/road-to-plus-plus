#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <istream>
#include <vector>

using std::complex;
using std::cout;
using std::int8_t;
using std::vector;

namespace pt = boost::property_tree;
struct Config {
  double cxmin;
  double cxmax;
  double cymin;
  double cymax;
  size_t max_iterations;
  size_t ixsize;
  size_t iysize;
  friend std::ostream &operator<<(std::ostream &stream, const Config &config) {
    return stream << "Config \n"
                  << "{ cxmin: " << config.cxmin << " ,cxmax: " << config.cxmax
                  << "}\n"
                  << "{ cymin: " << config.cymin << " ,cymax: " << config.cymax
                  << "}\n"
                  << " Number of iterations: " << config.max_iterations << "\n"
                  << "{ ixsize: " << config.iysize
                  << " ,iysize: " << config.ixsize << "}\n";
  }
};

class Mandelbrot {
  Config config;
  vector<vector<int8_t>> image;

public:
  Mandelbrot(Config config)
      : config(config), image(config.iysize, vector<int8_t>(config.ixsize, 0)) {
#ifdef DEBUG
    cout << config;
#endif
  }

  void compute() {
    double cxmin = config.cxmin;
    double cymin = config.cymin;
    double cxmax = config.cxmax;
    double cymax = config.cymax;

    const int width = config.iysize;
    const int height = config.ixsize;
    const int num_pixels = width * height;

    // const complex<double> center((cxmax - cxmin) / 2.0, (cymax - cymin) / 2.0);
    const complex<double> begin = complex<double>(cxmin, cymin);
    double xinc = (cxmax - cxmin) / width;
    double yinc = (cymax - cymin) / height;

#pragma omp parallel for schedule(dynamic)
    for (int pix = 0; pix < num_pixels; ++pix) {
      const int x = pix % width, y = pix / width;

      complex<double> c = begin + complex<double>(y*yinc, x * xinc);
      // cout <<  c << " " << x << " " << y << "\n" ;

      complex<double> z = c;
      size_t iteration = 0;
      for (; iteration < config.max_iterations; ++iteration) {
        if (std::abs(z) >= 2)
          break;
        z = z * z + c;
      }
      // cout << x << " " << y << " " << iteration << "\n";
      if (iteration == config.max_iterations)
        iteration = 4;

      {
        // TODO: Fix coloring
        if (iteration != 4) {
          image[x][y] = iteration % 4;
        }
      }
    }
  }

  void serialize() {
    for (auto row : image) {
      for (auto pixel : row) {
        cout << (pixel ? "*" : " ");
      }
      cout << "\n";
    }
  }

  // Loosely based on https://github.com/skeeto/mandel-simd/blob/master/mandel.c
  // TODO: Fix coloring
  void write_image() {
    uint8_t imgbuf[3 * config.iysize * config.ixsize];
    const uint32_t color[] = {0xf53d3d, 0xf53d3d, 0xf53d93, 0xf53d93, 0xffffff};
    const size_t width = config.iysize;
    const int height = config.ixsize;
    const int num_pixels = width * height;
    for (int pix = 0; pix < num_pixels; ++pix) {
      const int x = pix % width, y = pix / width;
        // printf("%zu %zu %d\n", x, y, image[x][y]);
        uint8_t pxlclr = color[image[x][y]];
        // cout << y * 3 * config.ixsize + 3 * x << "\n";
        imgbuf[x * 3 * config.ixsize + 3 * y + 0] = pxlclr >> 16;
        imgbuf[x * 3 * config.ixsize + 3 * y + 1] = pxlclr >> 8;
        imgbuf[x * 3 * config.ixsize + 3 * y + 2] = pxlclr >> 0;
    }
    fprintf(stdout, "P6\n%zu %zu\n%d\n", config.ixsize, config.iysize, 255);
    fwrite(imgbuf, sizeof(imgbuf), 1, stdout);
  }
};

int main(int argc, char **argv) {
  pt::ptree tree;
  pt::read_ini("./config.ini", tree);

  Mandelbrot mandel({
      .cxmin = tree.get<double>("xmin", -2.5),
      .cxmax = tree.get<double>("xmax", 1.5),

      .cymin = tree.get<double>("ymin", -2.5),
      .cymax = tree.get<double>("ymax", 1.5),

      .max_iterations = tree.get<size_t>("max_iterations", 256),

      .ixsize = tree.get<size_t>("ixsize", 1000),
      .iysize = tree.get<size_t>("iysize", 1000),
  });
  mandel.compute();
#ifdef DEBUG
  mandel.serialize();
#endif
  mandel.write_image();

  return 0;
}
