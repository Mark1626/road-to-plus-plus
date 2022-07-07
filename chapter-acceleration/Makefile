HIPIFY=hipify-perl
HIPCC=hipcc
NVCC=nvcc
FLAGS=-O3

%.cu: %.cpp
	$(HIPIFY) -o $@ $<

%-hip: %.cpp
	$(HIPCC) -o $@ $< $(FLAGS)

%-cuda: %.cpp
	$(NVCC) -o $@ $< $(FLAGS)
