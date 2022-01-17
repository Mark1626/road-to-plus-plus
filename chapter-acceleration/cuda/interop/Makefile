NVCC=nvcc
LIBS=`pkg-config --libs cuda-11.4 cudart-11.4`

lib.o: lib.cu lib.cuh
	$(NVCC) -c $<

interop.o: interop.cc
	$(CXX) -c $<

interop: interop.o lib.o
	$(CXX) -o $@ $^ $(LIBS)

