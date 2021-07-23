CXXFLAGS=-Wall -Wextra -std=c++11 -Ofast -Xpreprocessor -fopenmp -lomp

all: spawn-thread-11 spawn-pthread fork sse

allASM: spawn-thread-11-ASM spawn-pthread-ASM fork-ASM

fork: fork.cpp

spawn-thread-11: spawn-thread-11.cpp

spawn-pthread: spawn-pthread.cpp

sse: sse.cpp

fork-ASM: fork.cpp
	$(CXX) -S $^ $(CXXFLAGS)

spawn-thread-11-ASM: spawn-thread-11.cpp
	$(CXX) -S $^ $(CXXFLAGS)

spawn-pthread-ASM: spawn-pthread.cpp
	$(CXX) -S $^ $(CXXFLAGS)

clean:
	rm -rf *.s \
	spawn-thread-11 spawn-pthread fork

.PHONY: all clean
