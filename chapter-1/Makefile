CXXFLAGS=-Wall -Wextra -g

hello-world: hello-world.cpp

ctor: ctor.cpp

.PHONY: clean format

format:
	clang-format -i *.cpp

clean:
	rm hello-world
	rm ctor
