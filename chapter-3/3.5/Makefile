.ONESHELL=

MPICXX=mpic++

CXXFLAGS=-Wall -Wextra -O3 -fopenmp -lboost_serialization -lboost_mpi -std=c++14 -lbenchmark
PKG_CONFIG=`pkg-config --libs benchmark cppunit liblog4cxx`

all: tbench test_tdigest texample

SHAREDFLAGS=-fPIC -shared -Wl,-soname,libtdigest.so -I .

libtdigest.so: TDigest.cc TDigest.hh
	$(CXX) -o $@ $^ $(CXXFLAGS) $(PKG_CONFIG) $(SHAREDFLAGS)

TDigest.o: TDigest.hh

%: %.cc TDigest.hh
	$(CXX) -o $@ $^ $(CXXFLAGS) $(PKG_CONFIG)

clean:
	rm tbench texample test_tdigest

.PHONY: clean libtdigest.so
