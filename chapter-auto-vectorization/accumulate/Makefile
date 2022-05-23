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

accumulate-case1: accumulate-case1.cc
	$(CXX) -o $@ $< $(CXXFLAGS) `pkg-config --cflags --libs benchmark`

clean:
	@rm accumulate-case1

.PHONY: clean
