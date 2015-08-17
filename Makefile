export inc_path=$(PWD)/include
export libs_path=$(PWD)/libs
export src_path=$(PWD)/src

export components := anfis anfis/training cluster detail
export srcdir := $(PWD)/src
export bindir := $(PWD)/bin
export builddir := $(bindir)/.build
export srcdirs := $(srcdir) $(addprefix $(srcdir)/,$(components))
export sources := $(wildcard $(addsuffix /*.cpp,$(srcdirs)))
export objs := $(patsubst $(srcdir)/%,$(builddir)/%,$(patsubst %.cpp,%.o,$(sources)))

export CXXFLAGS+=-Wall -Wextra -ansi -pedantic
#export CXXFLAGS+=-Wall -Wextra -std=c++11 -pedantic -DFL_CPP11
export CXXFLAGS+=-g -Og
export CXXFLAGS+=-I$(inc_path)
#export CXXFLAGS+=-I$(libs_path)/boost/include
export CXXFLAGS+=-I$(HOME)/sys/src/git/boost
export CXXFLAGS+=-I$(HOME)/sys/src/git/fuzzylite/fuzzylite
#export CXXFLAGS+=-I$(HOME)/Projects/src/fuzzylite/fuzzylite
#export LDFLAGS+=-L$(HOME)/sys/src/git/boost/stage/libs
export LDFLAGS+=-L$(HOME)/sys/src/git/fuzzylite/fuzzylite/debug/bin -lfuzzylited
#export LDFLAGS+=-L$(HOME)/Projects/src/fuzzylite/fuzzylite/debug/bin -lfuzzylited
export LDFLAGS+=-lm
export CC=$(CXX)

.PHONY: all clean examples test

all: bin examples test

warning:
#	@echo "******************************************************"
#	@echo "******* Change the FUZZYLITE path when ready! ********"
#	@echo "******************************************************"
#	@echo ""

bin: warning
	cd bin && $(MAKE)

examples: warning bin
	cd examples && $(MAKE)

test: warning bin
	cd test && $(MAKE)

clean:
	cd bin && $(MAKE) clean
	cd test && $(MAKE) clean
	cd examples && $(MAKE) clean
