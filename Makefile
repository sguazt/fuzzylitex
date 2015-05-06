export inc_path=$(PWD)/include
export libs_path=$(PWD)/libs
export src_path=$(PWD)/src
export CXXFLAGS+=-Wall -Wextra -ansi -pedantic
#export CXXFLAGS+=-Wall -Wextra -std=c++11 -pedantic -DFL_CPP11
export CXXFLAGS+=-g -Og
export CXXFLAGS+=-I$(inc_path)
#export CXXFLAGS+=-I$(libs_path)/boost/include
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
