CXXFLAGS=-Wall -Wextra -Werror -g -std=c++11
MPFLAGS=-Xpreprocessor -fopenmp -lomp

ifneq ($(VERBOSE),)
CXXFLAGS:=$(CXXFLAGS) -DVERBOSE
endif

all: main omain pmain pomain

perlin.o: perlin.cpp perlin.hpp
	$(CXX) -c $^ $(CXXFLAGS)

test: perlin.o test.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

main: perlin.o main.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

omain: perlin.o main.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) -O3

pmain: perlin.o main.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) $(MPFLAGS)

pomain: perlin.o main.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) -O3 $(MPFLAGS)

clean:
	rm *.o *.gch main omain pmain pomain test

.PHONY: clean all
