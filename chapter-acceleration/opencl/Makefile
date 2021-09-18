CXXFLAGS:=-Wall -Wextra -std=c++14 -O3
OPENCLFLAG:=-framework OpenCL
FLAGS += $(CXXFLAGS) $(OPENCLFLAG)

all: vector_add_gpu vector_add_cpu gol_cpu gol_gpu cl_info atomic

cl_info: cl_info.cc
	$(CXX) -o $@ $< $(FLAGS)

vector_add_gpu: vector_add.cc vector_add.cl
	$(CXX) -o $@ $< -DGPU $(FLAGS)

vector_add_cpu: vector_add.cc vector_add.cl
	$(CXX) -o $@ $< $(FLAGS)

gol_cpu: gol.cc gol.cl
	$(CXX) -o $@ $< $(FLAGS) -DWITH_PPM

gol_gpu: gol.cc gol.cl
	$(CXX) -o $@ $< -DGPU $(FLAGS) -DWITH_PPM

atomic: atomic.cc atomic.cl
	$(CXX) -o $@ $< $(FLAGS)

sandpile: sandpile.cc sandpile.cl
	$(CXX) -o $@ $< $(FLAGS)

clean:
	$(RM) cl_info vector_add_cpu vector_add_gpu atomic *.o *.dSYM
