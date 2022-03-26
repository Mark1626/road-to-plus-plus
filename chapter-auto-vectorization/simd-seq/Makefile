CXXFLAGS=-Wall -Wextra -std=c++11 \
			-O3 -march=native \
			-Xpreprocessor -fopenmp -lomp

complex-dot/complex-dot: complex-dot/complex-dot.cc \
													complex-dot/base.cc complex-dot/complex2.cc \
													complex-dot/complex4.cc complex-dot/soa.cc
	$(CXX) -o $@ $< $(CXXFLAGS) `pkg-config --cflags --libs benchmark`

clean:
	@rm complex-dot/complex-dot

.PHONY: clean
