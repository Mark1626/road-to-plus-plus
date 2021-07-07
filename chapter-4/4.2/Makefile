CXXFLAGS=-Wall -Wextra -g -std=c++11 -O3 -fopenmp

%.asm: %.cpp
	$(CXX) -S $^ $(CXXFLAGS)

clean:
	rm -rf case-1
