.ONESHELL=

MPICXX=mpic++

CXXFLAGS=-Wall -Wextra -O3 -fopenmp -lboost_serialization -lboost_mpi -std=c++14
PKG_CONFIG=`pkg-config --libs benchmark liblog4cxx cppunit`

all: tbench test_tdigest texample

SHAREDFLAGS=-fPIC -shared -Wl,-soname,libtdigest.so -I .

libtdigest.so: TDigest.cc TDigest.hh
	$(CXX) $(CXXFLAGS) $(PKG_CONFIG) $(SHAREDFLAGS) -o $@ $^

TDigest.o: TDigest.hh

%: %.cc TDigest.o
	$(CXX) $(CXXFLAGS) $(PKG_CONFIG) -o $@ $^

clean:
	rm tbench texample test_tdigest

.PHONY: clean libtdigest.so
