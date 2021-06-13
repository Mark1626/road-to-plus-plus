CXXFLAGS=-Wall -Wextra -g -std=c++11 -O3

ifneq ($(SIZE),)
CXXFLAGS:=$(CXXFLAGS) -DSIZE=$(SIZE)
endif

ifneq ($(DEBUG),)
CXXFLAGS:=$(CXXFLAGS) -DDEBUG
endif

ifneq ($(WITH_SSE),)
CXXFLAGS:=$(CXXFLAGS) -DWITH_SSE
endif

ifneq ($(ANIMATE),)
CXXFLAGS:=$(CXXFLAGS) -DANIMATE
endif

all: main

sandpile.o: sandpile.cpp sandpile.hpp
	$(CXX) -c $^ $(CXXFLAGS)

main: main.cpp sandpile.o
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm -rf *.o *.gch *.dSYM main

.PHONY: clean
