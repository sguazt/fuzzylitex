inc_path=../include
src_path=../src
CXXFLAGS+=-Wall -Wextra -ansi -pedantic
CXXFLAGS+=-g -Og
CXXFLAGS+=-I../include
CXXFLAGS+=-I$(HOME)/sys/src/git/boost
CXXFLAGS+=-I$(HOME)/sys/src/git/fuzzylite/fuzzylite
#LDFLAGS+=-L$(HOME)/sys/src/git/boost/stage/libs
LDFLAGS+=-L$(HOME)/sys/src/git/fuzzylite/fuzzylite/debug/bin -lfuzzylited
LDFLAGS+=-lm
CC=$(CXX)

.PHONY: all clean

all: test_anfis test_ann

test_anfis: test_anfis.o engine.o nodes.o terms.o
	$(CXX) $(CXXFLAGS) -o test_anfis test_anfis.o engine.o nodes.o terms.o $(LDFLAGS)

test_anfis.o: test_anfis.cpp
	$(CXX) $(CXXFLAGS) -c -o test_anfis.o test_anfis.cpp

engine.o: $(src_path)/anfis/engine.cpp
	$(CXX) $(CXXFLAGS) -c -o engine.o $(src_path)/anfis/engine.cpp

nodes.o: $(src_path)/anfis/nodes.cpp
	$(CXX) $(CXXFLAGS) -c -o nodes.o $(src_path)/anfis/nodes.cpp

terms.o: $(src_path)/detail/terms.cpp
	$(CXX) $(CXXFLAGS) -c -o terms.o $(src_path)/detail/terms.cpp

training.o: $(src_path)/anfis/training.cpp
	$(CXX) $(CXXFLAGS) -c -o training.o $(src_path)/anfis/training.cpp

clean:
	rm -f *.o \
		  test_anfis \
		  test_ann
