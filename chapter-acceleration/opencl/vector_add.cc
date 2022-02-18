#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <cassert>
#include <string>
#include <vector>

using std::string;
using std::vector;

struct CLContext {
  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_uint num_devices;
  cl_uint num_platforms;
  cl_context context;
  CLContext() {
    platform_id = NULL;
    device_id = NULL;
    num_platforms = 0;
    num_devices = 0;
    context = NULL;
  }
  ~CLContext() {
    if (context) {
      cl_int ret = clReleaseContext(context);
      // std::cout << "Context released \n";
      if (ret != CL_SUCCESS) {
        std::cerr << "Some error releasing the context";
      }
    }
  }
};

struct KernelCtx {
  cl_command_queue command_queue;
  cl_kernel kernel;
  cl_program program;

  KernelCtx() {
    command_queue = NULL;
    kernel = NULL;
    program = NULL;
  }

  ~KernelCtx() {
    cl_int ret;
    if (command_queue) {
      ret = clFlush(command_queue);
      ret = clFinish(command_queue);
    }
    if (program) {
      // std::cout << "Program released \n";
      ret = clReleaseProgram(program);
      if (ret != CL_SUCCESS) {
        std::cerr << "Some error releasing the program";
      }
    }
    if (kernel) {
      // std::cout << "Kernel released \n";
      ret = clReleaseKernel(kernel);
      if (ret != CL_SUCCESS) {
        std::cerr << "Some error releasing the kernel";
      }
    }
    if (command_queue) {
      // std::cout << "Command Queue released \n";
      ret = clReleaseCommandQueue(command_queue);
      if (ret != CL_SUCCESS) {
        std::cerr << "Some error releasing the command queue";
      }
    }
  }
};

struct KernelMem {
  cl_mem mem;
  KernelMem(cl_context context, const int size, cl_int *ret) {
    mem = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(int), NULL,
                         ret);
  };
  ~KernelMem() {
    if (mem) {
      // std::cout << "Memory Released\n";
      cl_int ret = clReleaseMemObject(mem);
      if (ret != CL_SUCCESS) {
        std::cerr << "Some error releasing the memory";
      }
    }
  }
};

void init_context(CLContext *context) {
  cl_int ret =
      clGetPlatformIDs(1, &context->platform_id, &context->num_platforms);

  int device_type;
#ifdef GPU
  std::cout << "Using GPU\n";
  device_type = CL_DEVICE_TYPE_GPU;
#else
  std::cout << "Using CPU\n";
  device_type = CL_DEVICE_TYPE_CPU;
#endif

  ret = clGetDeviceIDs(context->platform_id, device_type, 1,
                       &context->device_id, &context->num_devices);

  context->context =
      clCreateContext(NULL, 1, &context->device_id, NULL, NULL, &ret);
}

int main() {
  const int LIST_SIZE = 1 << 24;

  vector<int> a(LIST_SIZE, 0);
  vector<int> b(LIST_SIZE, 0);
  vector<int> c(LIST_SIZE, 0);

  for (int i = 0; i < LIST_SIZE; i++) {
    a[i] = i;
    b[i] = LIST_SIZE - i;
  }

  std::fstream file("vector_add.cl");

  string prog(std::istreambuf_iterator<char>(file),
              (std::istreambuf_iterator<char>()));

  CLContext ctx;
  init_context(&ctx);

  KernelCtx kctx;

  // Push data to device
  cl_int ret;
  KernelMem a_mem(ctx.context, LIST_SIZE, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "Error creating memory";
    return EXIT_FAILURE;
  }
  KernelMem b_mem(ctx.context, LIST_SIZE, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "Error creating memory";
    return EXIT_FAILURE;
  }
  KernelMem c_mem(ctx.context, LIST_SIZE, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "Error creating memory";
    return EXIT_FAILURE;
  }
  // std::cout << prog.c_str();

  const char *prog_src = prog.c_str();
  const size_t prog_size = prog.length();
  kctx.program =
      clCreateProgramWithSource(ctx.context, 1, (const char **)&prog_src,
                                (const size_t *)&prog_size, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating program " << ret << "\n";
    return EXIT_FAILURE;
  }

  ret = clBuildProgram(kctx.program, 1, &ctx.device_id, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cout << "Error building program " << ret << "\n";
    return EXIT_FAILURE;
  }

  kctx.kernel = clCreateKernel(kctx.program, "vector_add", &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating kernel " << ret << " \n";
    return EXIT_FAILURE;
  }

  ret = clSetKernelArg(kctx.kernel, 0, sizeof(cl_mem), (void *)&a_mem.mem);
  ret |= clSetKernelArg(kctx.kernel, 1, sizeof(cl_mem), (void *)&b_mem.mem);
  ret |= clSetKernelArg(kctx.kernel, 2, sizeof(cl_mem), (void *)&c_mem.mem);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting argument\n";
    return EXIT_FAILURE;
  }

  kctx.command_queue =
      clCreateCommandQueue(ctx.context, ctx.device_id, 0, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating command queue " << ret << "\n";
    return EXIT_FAILURE;
  }

  clEnqueueWriteBuffer(kctx.command_queue, a_mem.mem, CL_TRUE, 0,
                       LIST_SIZE * sizeof(int), a.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(kctx.command_queue, b_mem.mem, CL_TRUE, 0,
                       LIST_SIZE * sizeof(int), b.data(), 0, NULL, NULL);

  size_t items = LIST_SIZE;
  size_t item_size = 64;

  ret = clEnqueueNDRangeKernel(kctx.command_queue, kctx.kernel, 1, NULL, &items,
                               &item_size, 0, NULL, NULL);

  ret = clEnqueueReadBuffer(kctx.command_queue, c_mem.mem, CL_TRUE, 0,
                            LIST_SIZE * sizeof(int), c.data(), 0, NULL, NULL);

  for (int i = 0; i < LIST_SIZE; ++i)
    assert(c[i] == LIST_SIZE);

  std::cout << "All assertions passed\n";
}
