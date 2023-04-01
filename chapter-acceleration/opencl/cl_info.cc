#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void printDeviceInfo(cl::Device d) {
  std::cout << "Device Name: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
  std::cout << "Device Version: " << d.getInfo<CL_DEVICE_VERSION>()
            << std::endl;
  std::cout << "Device Vendor: " << d.getInfo<CL_DEVICE_VENDOR>() << std::endl;
  std::cout << "Driver Version: " << d.getInfo<CL_DRIVER_VERSION>()
            << std::endl;
  std::cout << "Max compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
            << std::endl;
  std::cout << "Max Work Item Dimensions: "
            << d.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
  std::cout << "Max Work Group Size: "
            << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
  auto sizes = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  std::cout << "Max Work Item Sizes: " ;
  for (auto size : sizes)
    std::cout << size << " ";
  std::cout << std::endl;
}

int main() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    std::cout << "No platforms found";
    exit(1);
  }
  for (auto &p : platforms) {
    std::cout << "Platform name: " << p.getInfo<CL_PLATFORM_NAME>()
              << std::endl;
    std::cout << "Platform verison: " << p.getInfo<CL_PLATFORM_VERSION>()
              << std::endl;
    auto extensions = p.getInfo<CL_PLATFORM_EXTENSIONS>();
    std::cout << "Available extensions: \n";
    std::stringstream ss(extensions);
    std::string ext;
    while (std::getline(ss, ext, ' ')) {
      std::cout << ext << std::endl;
    }
    std::cout << std::endl;
  }

  cl::Platform default_platform = platforms[0];
  std::cout << "Using platform: "
            << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

  std::cout << "GPUs" << std::endl;
  std::cout << "-------------------------" << std::endl;
  std::vector<cl::Device> gpus;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
  for (auto &d : gpus) {
    printDeviceInfo(d);
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "CPUs" << std::endl;
  std::cout << "-------------------------" << std::endl;
  std::vector<cl::Device> cpus;
  default_platform.getDevices(CL_DEVICE_TYPE_CPU, &cpus);
  for (auto &d : cpus) {
    printDeviceInfo(d);
    std::cout << std::endl;
  }

  std::cout << "Default" << std::endl;
  std::cout << "-------------------------" << std::endl;
  std::vector<cl::Device> defaults;
  default_platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &defaults);
  for (auto &d : defaults) {
    printDeviceInfo(d);
    std::cout << std::endl;
  }
}