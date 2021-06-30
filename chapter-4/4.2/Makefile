CXXFLAGS=-Wall -Wextra -g -std=c++11 -Xpreprocessor -fopenmp -lomp

%.asm: %.cpp
	$(CXX) -S $^ $(CXXFLAGS)

clean:
	rm -rf case-1
