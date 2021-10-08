CXXFLAGS=-Wall -Wextra -O2
OPENACCFLAG:=-fopenacc

NVCXX=nvc++
NVCXXFLAGS=-mp -target=gpu

all: vector_add_cpu vector_add_gpu

vector_add_cpu: vector_add.cc
	$(CXX) -o $@ $< $(CXXFLAGS)

vector_add_gpu: vector_add.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(OPENACCFLAG)

collide_gpu:
	$(NVCXX) -o $@ $< $(NVCXXFLAGS)

clean:
	$(RM) vector_add_cpu vector_add_gpu collide_gpu

.PHONY: all clean
