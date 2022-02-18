CUDA=nvcc

all: hello vector_add class interop

hello: hello.cu
	$(CUDA) -o $@ $^

vector_add_1_1: vector_add.cu
	$(CUDA) -o $@ $^ -DTHREAD_BLOCKS=1 -DNUM_THREADS=1

vector_add_1_4: vector_add.cu
	$(CUDA) -o $@ $^ -DTHREAD_BLOCKS=1 -DNUM_THREADS=4

vector_add_4_1: vector_add.cu
	$(CUDA) -o $@ $^ -DTHREAD_BLOCKS=4 -DNUM_THREADS=1

vector_add_4_4: vector_add.cu
	$(CUDA) -o $@ $^ -DTHREAD_BLOCKS=4 -DNUM_THREADS=4

vector_add: vector_add_1_1 vector_add_1_4 vector_add_4_1 vector_add_4_4

class: class.cu
	$(CUDA) -o $@ $<

interop:
	make -C interop $@

clean:
	$(RM) hello vector_add_*

.PHONY: all clean vector_add interop

