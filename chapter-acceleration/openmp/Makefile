CXXFLAGS=-Wall -Wextra -O3
OPENMPFLAG:=-fopenmp
OFFLOADFLAGS:=-foffload=nvptx-none

CXX=g++

all: vector_add_cpu vector_add_gpu reduction_cpu reduction_gpu collide_cpu collide_gpu

vector_add_cpu: vector_add.cc
	$(CXX) -o $@ $< $(CXXFLAGS)

vector_add_gpu: vector_add.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(OPENMPFLAG) $(OFFLOADFLAGS)

reduction_cpu: reduction.cc
	$(CXX) -o $@ $< $(CXXFLAGS)

reduction_gpu: reduction.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(OPENMPFLAG) $(OFFLOADFLAGS)

collide_cpu: collide_gpu.cc
	$(CXX) -o $@ $< $(CXXFLAGS)

collide_gpu: collide_gpu.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(OPENMPFLAG) $(OFFLOADFLAGS)

collide: collide.c
	$(CC) -o $@ $< $(CXXFLAGS) $(OPENMPFLAG)

offload-case-studies/offload-case-studies: offload-case-studies/offload-case-studies.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(OPENMPFLAG) $(OFFLOADFLAGS)

clean:
	$(RM) vector_add_cpu vector_add_gpu reduction_cpu reduction_gpu collide_cpu collide_gpu

.PHONY: all clean
