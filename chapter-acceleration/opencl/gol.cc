#include <OpenCL/opencl.h>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define WITH_PPM
// #define GPU

#ifdef WITH_PPM
#include <fstream>
#include <ppm/ppm.hpp>
#endif

using std::string;

void check_status(cl_int ret, std::string msg) {
  if (ret != CL_SUCCESS) {
    std::cerr << msg << ret;
    exit(1);
  }
}

class CLContext {
public:
  cl_platform_id platform_id;
  cl_device_id device_id;

  cl_uint num_devices;
  cl_uint num_platforms;

  cl_context context;
  cl_command_queue command_queue;

  CLContext() {
    command_queue = NULL;
    platform_id = NULL;
    device_id = NULL;
    num_platforms = 0;
    num_devices = 0;
    context = NULL;
  }
  ~CLContext() {
    cl_int ret;
    if (command_queue) {
      ret = clFlush(command_queue);
      ret = clFinish(command_queue);
    }
    if (context) {
      ret = clReleaseContext(context);
      // std::cout << "Context released \n";
      if (ret != CL_SUCCESS) {
        std::cerr << "Some error releasing the context";
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

  cl_int init_context() {
    cl_int ret = clGetPlatformIDs(1, &platform_id, &num_platforms);

    int device_type;
#ifdef GPU
    std::cout << "Using GPU\n";
    device_type = CL_DEVICE_TYPE_GPU;
#else
    std::cout << "Using CPU\n";
    device_type = CL_DEVICE_TYPE_CPU;
#endif

    ret = clGetDeviceIDs(platform_id, device_type, 1, &device_id, &num_devices);
    check_status(ret, "Error getting device id");

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    check_status(ret, "Error creating context");

    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    check_status(ret, "Error creating command queue");

    return CL_SUCCESS;
  }
};

struct KernelCtx {
  cl_kernel kernel;
  cl_program program;

public:
  KernelCtx() {
    kernel = NULL;
    program = NULL;
  }

  ~KernelCtx() {
    cl_int ret;

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
  }

  void create_program_from_source(cl_device_id *device_id, cl_context *context,
                                  string source, string kernel_name) {
    std::fstream file(source);
    string prog(std::istreambuf_iterator<char>(file),
                (std::istreambuf_iterator<char>()));

    const char *prog_src = prog.c_str();
    size_t prog_size = prog.length();

    cl_int ret;
    program = clCreateProgramWithSource(*context, 1, (const char **)&prog_src,
                                        (const size_t *)&prog_size, &ret);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error creating program " << ret << "\n";
      exit(1);
    }

    ret = clBuildProgram(program, 1, device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error building program " << ret << "\n";
      size_t len;
      char buffer[2048];
      clGetProgramBuildInfo(program, *device_id, CL_PROGRAM_BUILD_LOG,
                            sizeof(buffer), (void *)buffer, &len);
      std::cerr << buffer;
      exit(1);
    }

    kernel = clCreateKernel(program, kernel_name.c_str(), &ret);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error creating kernel " << ret << " \n";
      exit(1);
    }
  }
};

struct KernelMem {
  cl_mem mem;
  KernelMem() { mem = NULL; }
  KernelMem(cl_context context, const int size, cl_int *ret) {
    mem = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(bool), NULL,
                         ret);
  };
  ~KernelMem() {
    // std::cout << "Destructor called";
    if (mem) {
      // std::cout << "Memory Released\n";
      cl_int ret = clReleaseMemObject(mem);
      if (ret != CL_SUCCESS) {
        std::cerr << "Some error releasing the memory" << ret;
      }
    }
  }
};

class Board {
  CLContext ctx;
  KernelCtx kctx;
  KernelMem *input;
  KernelMem *output;

  int width;
  int height;
#ifdef WITH_PPM
  PPM ppm;
#endif
  const size_t board_size;
  bool *board;

public:
  Board(int width, int height)
      : width(width), height(height),
#ifdef WITH_PPM
        ppm(width, height),
#endif
        board_size(width * height) {
    board = new bool[board_size];
    init_board();
    ctx.init_context();
    kctx.create_program_from_source(&ctx.device_id, &ctx.context, "gol.cl",
                                    "cl_gol");
  }

  ~Board() {
    if (board) {
      delete[] board;
    }
    delete input;
    delete output;
  }

  void init_board() {
    for (unsigned int i = 0; i < board_size; i++)
      board[i] = random() > (INT_MAX / 2);
  }

  void prepare_kernel() {
    cl_int ret;
    input = new KernelMem(ctx.context, board_size, &ret);
    check_status(ret, "Error creating memory");
    output = new KernelMem(ctx.context, board_size, &ret);
    check_status(ret, "Error creating memory");

    // input = inp;
    // output = out;

    ret = 0;
    ret |= clSetKernelArg(kctx.kernel, 0, sizeof(cl_mem), &input->mem);
    check_status(ret, "Error setting arguments0 ");
    ret |= clSetKernelArg(kctx.kernel, 1, sizeof(cl_mem), &output->mem);
    check_status(ret, "Error setting arguments1");
    ret |= clSetKernelArg(kctx.kernel, 2, sizeof(int), (void *)&width);
    check_status(ret, "Error setting arguments 2");
    ret |= clSetKernelArg(kctx.kernel, 3, sizeof(int), (void *)&height);
    check_status(ret, "Error setting arguments");
  }

  void run_game(int iterations) {
    if (iterations == 0)
      return;

    size_t workgroup_size;
    cl_int ret = clGetKernelWorkGroupInfo(
        kctx.kernel, ctx.device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
        &workgroup_size, NULL);
    check_status(ret, "Error creating worker group");

    ret = clEnqueueWriteBuffer(ctx.command_queue, input->mem, CL_TRUE, 0,
                               board_size * sizeof(bool), board, 0, NULL, NULL);
    check_status(ret, "Failed to enqueue data");

    for (int i = 0; i < iterations; i++) {
      ret = clEnqueueNDRangeKernel(ctx.command_queue, kctx.kernel, 1, NULL,
                                   &board_size, &workgroup_size, 0, NULL, NULL);
      check_status(ret, "Unable to unqueue kernel");
      if (i < iterations - 1) {
        ret = clEnqueueCopyBuffer(ctx.command_queue, output->mem, input->mem, 0,
                                  0, board_size * sizeof(bool), 0, NULL, NULL);
        check_status(ret, "Unable to copy");
      }
    }
    ret = clEnqueueReadBuffer(ctx.command_queue, output->mem, CL_TRUE, 0,
                              board_size * sizeof(bool), board, 0, NULL, NULL);
    check_status(ret, "Unable to read result");
  }

  void print(bool first) {
    int i = 0;
    for (int y = 0; y < height; y++) {

      for (int x = 0; x < width; x++) {
#ifdef WITH_PPM
        board[i] ? ppm.set(x, y, RGB(255, 255, 255))
                 : ppm.set(x, y, RGB(0, 0, 0));

#else
        std::cout << (board[i] ? "*" : " ");
#endif
        i++;
      }
#ifndef WITH_PPM
      std::cout << "\n";
#endif
    }
#ifdef WITH_PPM
    if (first) {

      std::fstream file("first.ppm", std::ios::out);
      ppm.write(file);
    } else {
      std::fstream file("last.ppm", std::ios::out);
      ppm.write(file);
    }
#endif
  }
};

int main() {
  Board board(1280, 640);
  board.prepare_kernel();
  board.print(true);

  board.run_game(10000);
  board.print(false);
}
