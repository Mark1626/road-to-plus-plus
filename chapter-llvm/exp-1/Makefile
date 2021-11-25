LLVM_CONFIG=llvm-config-10
OPT=opt-10
LLVM_FLAGS=$(shell $(LLVM_CONFIG) --libs core native --cxxflags -ldflags)
CC=clang

count-pass.so: count-pass.cc
	$(CC) -shared -fPIC $^ -o $@ $(LLVM_FLAGS)

hello.bc: hello.cc
	$(CC) -O3 -emit-llvm $^ -c -o $@

run-pass: count-pass.so hello.bc
	opt-10 -load ./$< -count < hello.bc > /dev/null

run-pass-time: count-pass.so hello.bc
	opt-10 -load ./$< -count -time-passes < hello.bc > /dev/null

print:
	@echo "Hello World"

clean:
	@rm -f *.so *.bc *.o

.PHONY: print clean
