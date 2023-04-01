#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <immintrin.h>

#include "ppm.h"

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

using std::array;
using std::fstream;
using std::string;
using std::vector;

static const int x = 128;
static const int width = x + 2;
static const int height = x + 2;
static const int buff_size = width * height;

typedef cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer &,
                        const cl_int &, const cl_int &>
    sandpile_fn;

void stabilize(cl::Context &context, cl::CommandQueue &queue,
               std::array<cl_uchar, buff_size> &input,
               std::array<cl_uchar, buff_size> &output,
               sandpile_fn &cl_stabilize);

void serialize(unsigned char *buff, FILE *stream);

int main() {
  vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    std::cout << "No platforms found";
    exit(1);
  }
  cl::Platform default_platform = platforms[0];
  std::cout << "Using platform: "
            << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
  vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
  cl::Device device = all_devices[1];
  std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

  cl::Context context({device});

  fstream file("sandpile.cl");
  string kernel_code((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
  cl::Program::Sources sources;
  sources.push_back({kernel_code.c_str(), kernel_code.length()});

  cl::Program program(context, sources);
  if (program.build({device}) != CL_SUCCESS) {
    std::cout << "Error building "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
              << std::endl;
    exit(1);
  }

  std::array<cl_uchar, buff_size> input;
  std::array<cl_uchar, buff_size> output;

  cl::CommandQueue queue(context, device);
  cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer &, const cl_int &,
                  const cl_int &>
      cl_stabilize(cl::Kernel(program, "cl_stabilize"));

  int i = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x == 0 || y == 0)
        input[i] = 0;
      else if (x == height - 1 || y == width - 1)
        input[i] = 0;
      else
        input[i] = 6;
      i++;
    }
  }

  stabilize(context, queue, input, output, cl_stabilize);

  i = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x == 0 || y == 0)
        input[i] = 0;
      else if (x == height - 1 || y == width - 1)
        input[i] = 0;
      else
        input[i] = 6 - input[i];
      i++;
    }
  }

  // Print
  // i = 0;
  // for (int y = 0; y < height; y++) {
  //   for (int x = 0; x < width; x++) {
  //     std::cout << (int)input[i] << "";
  //     i++;
  //   }
  //   std::cout << std::endl;
  // }

  stabilize(context, queue, input, output, cl_stabilize);

  // Print
  // i = 0;
  // for (int y = 0; y < height; y++) {
  //   for (int x = 0; x < width; x++) {
  //     std::cout << (int)input[i] << "";
  //     i++;
  //   }
  //   std::cout << std::endl;
  // }

  FILE *outfile = fopen("sandpile.ppm", "w");
  serialize(input.data(), outfile);
  fclose(outfile);
}

void stabilize(cl::Context &context, cl::CommandQueue &queue,
               std::array<cl_uchar, buff_size> &input,
               std::array<cl_uchar, buff_size> &output,
               sandpile_fn &cl_stabilize) {
  cl_int ret;

  cl::Buffer buffer_inp(context, CL_MEM_READ_WRITE,
                        sizeof(cl_uchar) * buff_size);
  cl::Buffer buffer_out(context, CL_MEM_READ_WRITE,
                        sizeof(cl_uchar) * buff_size);
  cl::Buffer buffer_spill(context, CL_MEM_READ_WRITE, sizeof(int));

  int spills = 0;
  queue.enqueueWriteBuffer(buffer_inp, CL_TRUE, 0, sizeof(cl_uchar) * buff_size,
                           input.data());
  queue.enqueueWriteBuffer(buffer_out, CL_TRUE, 0, sizeof(cl_uchar) * buff_size,
                           output.data());

  cl::EnqueueArgs args(queue, cl::NullRange, cl::NDRange(buff_size),
                       cl::NullRange);

  do {
    spills = 0;
    ret = queue.enqueueWriteBuffer(buffer_spill, CL_TRUE, 0, sizeof(int),
                                   &spills);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error writing buffer";
      exit(1);
    }
    cl_stabilize(args, buffer_inp, buffer_out, buffer_spill, width, height)
        .wait();
    ret = queue.enqueueCopyBuffer(buffer_out, buffer_inp, 0, 0, buff_size);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error copying buffer";
      exit(1);
    }
    ret =
        queue.enqueueReadBuffer(buffer_spill, CL_TRUE, 0, sizeof(int), &spills);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error reading buffer";
      exit(1);
    }
    // std::cout << "Spills: " << spills << std::endl;
  } while (spills != 0);

  ret = queue.enqueueReadBuffer(buffer_inp, CL_TRUE, 0,
                                sizeof(cl_uchar) * buff_size, input.data());
  if (ret != CL_SUCCESS) {
    std::cerr << "Error reading buffer";
    exit(1);
  }
}

void serialize(unsigned char *buff, FILE *stream) {

  int w = width - 2;
  int h = height - 2;
  ppm img = ppm_init(w, h);
  RGB colors[] = {{255, 255, 255}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {0, 0, 0}};

  int idx = 0;
  for (std::int32_t j = 0; j < height; ++j) {
    for (std::int32_t i = 0; i < width; ++i) {
      if (!(i == 0 || j == 0 || i == width - 1 || j == height - 1)) {
        int sand = buff[idx];
        sand = (sand < 0 || sand > 3) ? 4 : sand;
        ppm_set(&img, i - 1, j - 1, colors[sand]);
      }
      idx++;
    }
  }

  ppm_serialize(&img, stream);
  // const uint32_t color[] = {0xf53d3d, 0x3d93f5, 0xf53d93, 0x99f53d};
}
