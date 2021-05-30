CXXFLAGS = -Wall -Wextra -std=c++11 -g
OBJS = sha1.o main.o

sha1.o: sha1.hpp sha1.cpp
	$(CXX) -c sha1.hpp sha1.cpp $(CXXFLAGS)

main.o: sha1.o main.cpp 
	$(CXX) -c main.cpp sha1.o $(CXXFLAGS)

main: $(OBJS) main.cpp
	$(CXX) -o $@ $(OBJS) $(CXXFLAGS)

.PHONY: clean

clean:
	rm *.o *.gch
