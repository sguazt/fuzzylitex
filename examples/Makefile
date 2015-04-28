inc_path=../include
src_path=../src
CXXFLAGS+=-Wall -Wextra -ansi -pedantic
CXXFLAGS+=-g -Og
CXXFLAGS+=-I$(inc_path)
CXXFLAGS+=-I$(HOME)/sys/src/git/boost
CXXFLAGS+=-I$(HOME)/sys/src/git/fuzzylite/fuzzylite
#LDFLAGS+=-L$(HOME)/sys/src/git/boost/stage/libs
LDFLAGS+=-L$(HOME)/sys/src/git/fuzzylite/fuzzylite/debug/bin -lfuzzylited
LDFLAGS+=-lm
CC=$(CXX)

.PHONY: all clean

all: anfis_invkinematics anfis_mackeyglass canfis_invkinematics

anfis_invkinematics: anfis_invkinematics.o engine.o nodes.o terms.o training.o
	$(CXX) $(CXXFLAGS) -o anfis_invkinematics anfis_invkinematics.o engine.o nodes.o terms.o training.o $(LDFLAGS)

anfis_mackeyglass: anfis_mackeyglass.o engine.o nodes.o terms.o training.o
	$(CXX) $(CXXFLAGS) -o anfis_mackeyglass anfis_mackeyglass.o engine.o nodes.o terms.o training.o $(LDFLAGS)

anfis_invkinematics.o: anfis_invkinematics.cpp
	$(CXX) $(CXXFLAGS) -c -o anfis_invkinematics.o anfis_invkinematics.cpp

anfis_mackeyglass.o: anfis_mackeyglass.cpp
	$(CXX) $(CXXFLAGS) -c -o anfis_mackeyglass.o anfis_mackeyglass.cpp

canfis_invkinematics: canfis_invkinematics.o engine.o nodes.o terms.o training.o
	$(CXX) $(CXXFLAGS) -o canfis_invkinematics canfis_invkinematics.o engine.o nodes.o terms.o training.o $(LDFLAGS)

canfis_invkinematics.o: canfis_invkinematics.cpp
	$(CXX) $(CXXFLAGS) -c -o canfis_invkinematics.o canfis_invkinematics.cpp

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
		  anfis_invkinematics \
		  anfis_mackeyglass \
		  canfis_invkinematics
