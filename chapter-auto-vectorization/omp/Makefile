UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
		OMPFLAGS += -fopenmp
endif
ifeq ($(UNAME_S),Darwin)
		OMPFLAGS += -Xpreprocessor -fopenmp -lomp
endif

CXXFLAGS=-O3 -march=native -Xpreprocessor -fopenmp -std=c++11
LIBS=pkg-config --cflags --libs benchmark

unroll/unroll: unroll/unroll.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(OMPFLAGS) -L/usr/local/lib -lbenchmark

reduction/reduction: reduction/reduction.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(OMPFLAGS) -L/usr/local/lib -lbenchmark

