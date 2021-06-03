CXXFLAGS = -Wall -Wextra -std=c++11 -g
HEADERS = life.hpp gol.hpp
OBJS = gol.o main.o

gol.o: gol.cpp $(HEADERS)
	$(CXX) -c gol.cpp gol.hpp life.hpp $(CXXFLAGS)

main: $(OBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

.PHONY: clean

clean:
	rm *.o main
