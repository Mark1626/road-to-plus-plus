CXXFLAGS=-Wall -Wextra -Ofast
OPENACCFLAG:=-fopenacc

all: vector_add

vector_add_cpu: vector_add.cc
	$(CXX) -o $@ $< $(CXXFLAGS)

vector_add_omp: vector_add.cc
	$(CXX) -o $@ $< $(CXXFLAGS) -DOMP

vector_add_gpu: vector_add.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(OPENACCFLAG)

clean:
	$(RM) vector_add_cpu vector_add_gpu vector_add_omp

.PHONY: all clean
