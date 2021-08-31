CXXFLAGS:=-Wall -Wextra -std=c++14 -framework OpenCL

all: vector_add_gpu vector_add_cpu gol_cpu gol_gpu

vector_add_gpu: vector_add.cc vector_add.cl
	$(CXX) -o $@ $< -DGPU $(CXXFLAGS)

vector_add_cpu: vector_add.cc vector_add.cl
	$(CXX) -o $@ $< $(CXXFLAGS)

gol_cpu: gol.cc gol.cl
	$(CXX) -o $@ $< $(CXXFLAGS)

gol_gpu: gol.cc gol.cl
	$(CXX) -o $@ $< -DGPU $(CXXFLAGS)

clean:
	$(RM) -f vector_add_cpu vector_add_gpu gpu-info *.o
