#inc_path=../include
#src_path=../src
#CXXFLAGS+=-Wall -Wextra -ansi -pedantic
#CXXFLAGS+=-g -Og
#CXXFLAGS+=-I../include
#CXXFLAGS+=-I$(HOME)/sys/src/git/boost
#CXXFLAGS+=-I$(HOME)/sys/src/git/fuzzylite/fuzzylite
##LDFLAGS+=-L$(HOME)/sys/src/git/boost/stage/libs
#LDFLAGS+=-L$(HOME)/sys/src/git/fuzzylite/fuzzylite/debug/bin -lfuzzylited
#LDFLAGS+=-lm
#CC=$(CXX)

.PHONY: all clean

all: test_anfis test_cluster_subtractive test_ann test_lsq

test_anfis: test_anfis.o engine.o nodes.o terms.o
	$(CXX) $(CXXFLAGS) -o test_anfis test_anfis.o engine.o nodes.o terms.o $(LDFLAGS)

test_anfis.o: test_anfis.cpp
	$(CXX) $(CXXFLAGS) -c -o test_anfis.o test_anfis.cpp

test_cluster_subtractive: test_cluster_subtractive.o subtractive.o
	$(CXX) $(CXXFLAGS) -o test_cluster_subtractive test_cluster_subtractive.o subtractive.o $(LDFLAGS)

test_lsq: test_lsq.o
	$(CXX) $(CXXFLAGS) -o test_lsq test_lsq.o $(LDFLAGS)

engine.o: $(src_path)/anfis/engine.cpp
	$(CXX) $(CXXFLAGS) -c -o engine.o $(src_path)/anfis/engine.cpp

nodes.o: $(src_path)/anfis/nodes.cpp
	$(CXX) $(CXXFLAGS) -c -o nodes.o $(src_path)/anfis/nodes.cpp

subtractive.o: $(src_path)/cluster/subtractive.cpp
	$(CXX) $(CXXFLAGS) -c -o subtractive.o $(src_path)/cluster/subtractive.cpp

terms.o: $(src_path)/detail/terms.cpp
	$(CXX) $(CXXFLAGS) -c -o terms.o $(src_path)/detail/terms.cpp

training.o: $(src_path)/anfis/training.cpp
	$(CXX) $(CXXFLAGS) -c -o training.o $(src_path)/anfis/training.cpp

clean:
	rm -f *.o \
		  test_anfis \
		  test_ann \
		  test_cluster_subtractive \
		  test_lsq
