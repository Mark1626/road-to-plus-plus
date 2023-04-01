#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

using std::fstream;
using std::string;
using std::vector;

const int ARRAY_SIZE = 1000000;

int main() {
  vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  for (auto &p : platforms) {
        std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
        std::cerr << "Plat: " << platver << "\n";
    }

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
  std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>()
            << "\n";

  cl::Context context({device});

  fstream file("atomic.cl");
  string kernel_code((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());

  cl::Program::Sources sources;
  sources.push_back({kernel_code.c_str(), kernel_code.length()});

  cl::Program program(context, sources);
  if (program.build({device}) != CL_SUCCESS) {
    std::cout << "Error building "<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
    exit(1);
  }

  cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(int) * ARRAY_SIZE);
  cl::Buffer bufferSum(context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong));
  vector<int> A(ARRAY_SIZE, 0);
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    A[i] = i;
  }
  cl_ulong sum = 0;

  cl::CommandQueue queue(context, device);
  queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, A.data());
  queue.enqueueWriteBuffer(bufferSum, CL_TRUE, 0, sizeof(cl_ulong), &sum);

  cl::make_kernel<cl::Buffer&, cl::Buffer&> atomic_sum(cl::Kernel(program, "atomic_sum"));
  cl::EnqueueArgs args(queue, cl::NullRange, cl::NDRange(ARRAY_SIZE), cl::NullRange);
  atomic_sum(args, bufferA, bufferSum).wait();

  queue.enqueueReadBuffer(bufferSum, CL_TRUE, 0, sizeof(cl_ulong), &sum);

  std::cout << "Sum: " << sum << std::endl;
}
