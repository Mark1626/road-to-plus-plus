CXXFLAGS=-Wall -Wextra -O3 -g -fno-omit-frame-pointer -fopenmp -fsanitize=address

.ONESHELL=

all: leak sandpile

leak: leak.cc

sandpile: sandpile.cc

clean:
	$(RM) leak sandpile

.PHONY: all
