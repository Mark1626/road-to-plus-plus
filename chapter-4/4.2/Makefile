CXXFLAGS=-Wall -Wextra -std=c++11 -Ofast -fopenmp

all: case-1 case-2 case-3

%.asm: %.cpp
	$(CXX) -S $^ $(CXXFLAGS)

clean:
	rm -rf case-1 case-2 case-3
