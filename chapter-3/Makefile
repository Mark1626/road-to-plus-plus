.ONESHELL=

CXXFLAGS=-Wall -Wextra -std=c++20 -O3
INCLUDES=-I /usr/local/include -I ./lib/
LIBS=-L/usr/local/lib

OUTDIR:=output

all: outdir 3.1 3.2 3.3

outdir:
	mkdir -p output

3.1:
	$(MAKE) -C 3.1
	mv 3.1/main output/sha1

3.2: 3.2/convolute.cc
	$(CXX) -o output/convolute $^ $(CXXFLAGS)

3.3: 3.3/hist_equ.cc
	$(CXX) -o output/hist_equ $^ $(CXXFLAGS) $(INCLUDES)

3.4: 3.4/dft.cc
	$(CXX) -o output/dft $^ $(CXXFLAGS) $(INCLUDES) $(LIBS) -g -lfftw3f

3.5: 3.5/parse.cc
	$(CXX) -o output/parse $^ $(CXXFLAGS) $(INCLUDES) $(LIBS) -g -lfftw3f

.PHONY: outdir 3.1 3.2 3.4 3.5
