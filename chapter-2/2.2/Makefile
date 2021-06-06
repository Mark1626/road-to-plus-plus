CXXFLAGS=-Wall -Wextra -g -std=c++11 -O3
MPFLAGS=-Xpreprocessor -fopenmp -lomp

ifneq ($(SIZE),)
CXXFLAGS:=$(CXXFLAGS) -DSIZE=$(SIZE)
endif

ifneq ($(DEBUG),)
CXXFLAGS:=$(CXXFLAGS) -DDEBUG
endif

ifneq ($(WITH_SSE),)
CXXFLAGS:=$(CXXFLAGS) -DWITH_SSE
endif

ifneq ($(WITH_MP),)
CXXFLAGS:=$(CXXFLAGS) $(MPFLAGS)
endif

gol.o: gol.cpp gol.hpp
	$(CXX) -c $^ $(CXXFLAGS)

main: gol.o main.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

all: main omain

clean:
	rm -rf *.o *.dSYM main

.PHONY: clean
