export inc_path=$(PWD)/include
export src_path=$(PWD)/src
export CXXFLAGS+=-Wall -Wextra -ansi -pedantic
export CXXFLAGS+=-g -Og
export CXXFLAGS+=-I$(inc_path)
export CXXFLAGS+=-I$(HOME)/sys/src/git/boost
export CXXFLAGS+=-I$(HOME)/sys/src/git/fuzzylite/fuzzylite
#export LDFLAGS+=-L$(HOME)/sys/src/git/boost/stage/libs
export LDFLAGS+=-L$(HOME)/sys/src/git/fuzzylite/fuzzylite/debug/bin -lfuzzylited
export LDFLAGS+=-lm
export CC=$(CXX)

.PHONY: all clean examples test

all: examples test

test:
	cd test && $(MAKE)

examples:
	cd examples && $(MAKE)

clean:
	cd test && $(MAKE) clean
	cd examples && $(MAKE) clean