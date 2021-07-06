CXXFLAGS=-Wall -Wextra -O3 -g -std=c++11 -isystem /usr/local/include -L/usr/local/lib
MPFLAGS=-Xpreprocessor -fopenmp -lomp

ifneq ($(VERBOSE),)
CXXFLAGS:=$(CXXFLAGS) -DVERBOSE
endif

all: main omain pmain pomain

perlin.o: perlin.cpp perlin.hpp
	$(CXX) -c $^ $(CXXFLAGS)

test: perlin.o test.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

main.o: perlin.o main.cpp
	$(CXX) -c $^ $(CXXFLAGS) $(MPFLAGS)

main: main.o
	$(CXX) -o main perlin.o main.o $(CXXFLAGS) $(MPFLAGS)

clean:
	rm *.o *.gch *.o main omain pmain pomain test

.PHONY: clean all
