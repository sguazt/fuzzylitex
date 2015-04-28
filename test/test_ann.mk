#fuzzylite_home=$(HOME)/Projects/src/fuzzylite
fuzzylite_home=$(HOME)/sys/src/git/fuzzylite

CXXFLAGS+=-Wall -Wextra -ansi -pedantic
CXXFLAGS+=-g -Og
LDFLAGS+=-lm
CXXFLAGS+=-I../..
CXXFLAGS+=-I$(HOME)/sys/src/git/boost
#LDFLAGS+=-L$(HOME)/sys/src/git/boost/stage/libs
CXXFLAGS+=-I$(fuzzylite_home)/fuzzylite
LDFLAGS+=-L$(fuzzylite_home)/fuzzylite/debug/bin -lfuzzylited
CC=$(CXX)

all: test_ann

clean:
	rm -f test_ann test_ann.o
