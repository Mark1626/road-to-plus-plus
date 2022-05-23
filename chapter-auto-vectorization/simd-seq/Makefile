UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
		OMPFLAGS += -fopenmp
endif
ifeq ($(UNAME_S),Darwin)
		OMPFLAGS += -Xpreprocessor -fopenmp -lomp
endif

CXXFLAGS=-Wall -Wextra -std=c++11 \
			-O3 -march=native -fcx-limited-range \
			$(OMPFLAGS) -g -fno-omit-frame-pointer -DBENCH -pthread

complex-dot/complex-dot: complex-dot/complex-dot.cc \
													complex-dot/base.cc complex-dot/complex2.cc \
													complex-dot/complex4.cc complex-dot/soa.cc \
													complex-dot/complex4-simd.cc complex-dot/complex2-simd.cc
	$(CXX) -o $@ $< $(CXXFLAGS) `pkg-config --cflags --libs benchmark`

grid/grid: grid/grid.cc grid/base.cc \
					grid/ptr.cc grid/complex2-simd.cc \
					grid/complex2-simd.cc grid/complex-simd-v2.cc \
					grid/complex-simd-v3.cc
	$(CXX) -o $@ $< $(CXXFLAGS) `pkg-config --cflags --libs benchmark` -DMILLI

clean:
	@rm complex-dot/complex-dot

.PHONY: clean
