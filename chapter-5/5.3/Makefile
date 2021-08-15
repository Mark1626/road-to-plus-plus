CXXFLAGS=-Wall -Wextra -g -O3 -fno-omit-frame-pointer

ifneq ($(ANIMATE),)
CXXFLAGS:=$(CXXFLAGS) -DANIMATE
endif

all: sandpile sandpile-p

sandpile: sandpile.cc
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm sandpile

.PHONY: clean all
