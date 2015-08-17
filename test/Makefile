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

all: test_anfis test_cluster_subtractive test_ann test_lsq test_rls

#test_anfis: test_anfis.o engine.o nodes.o terms.o
#	$(CXX) $(CXXFLAGS) -o test_anfis test_anfis.o engine.o nodes.o terms.o $(LDFLAGS)

test_anfis: test_anfis.o
	$(CXX) $(CXXFLAGS) -o test_anfis test_anfis.o $(LDFLAGS) -L$(bindir) -lfuzzylitex

#test_anfis.o: test_anfis.cpp
#	$(CXX) $(CXXFLAGS) -c -o test_anfis.o test_anfis.cpp

test_cluster_subtractive: test_cluster_subtractive.o
	$(CXX) $(CXXFLAGS) -o test_cluster_subtractive test_cluster_subtractive.o $(LDFLAGS) -L$(bindir) -lfuzzylitex

test_lsq: test_lsq.o
	$(CXX) $(CXXFLAGS) -o test_lsq test_lsq.o $(LDFLAGS) -L$(bindir) -lfuzzylitex

test_rls: test_rls.o
	$(CXX) $(CXXFLAGS) -o test_rls test_rls.o $(LDFLAGS) -L$(bindir) -lfuzzylitex

clean:
	rm -f *.o \
		  test_anfis \
		  test_ann \
		  test_cluster_subtractive \
		  test_lsq \
		  test_rls
