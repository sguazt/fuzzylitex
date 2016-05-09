##############################################
######## User-configurable parameters ########
##############################################
flx_have_lapack=1
##############################################


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

CXXFLAGS+=-Wall -Wextra -ansi -pedantic
#CXXFLAGS+=-Wall -Wextra -std=c++11 -pedantic -DFL_CPP11
CXXFLAGS+=-g -Og
CXXFLAGS+=-I$(inc_path)
#CXXFLAGS+=-I$(libs_path)/boost/include
CXXFLAGS+=-I$(HOME)/sys/src/git/boost
CXXFLAGS+=-I$(HOME)/sys/src/git/fuzzylite/fuzzylite
#CXXFLAGS+=-I$(HOME)/Projects/src/fuzzylite/fuzzylite
#LDFLAGS+=-L$(HOME)/sys/src/git/boost/stage/libs
LDFLAGS+=-L$(HOME)/sys/src/git/fuzzylite/fuzzylite/release/bin -lfuzzylite
#LDFLAGS+=-L$(HOME)/Projects/src/fuzzylite/fuzzylite/debug/bin -lfuzzylited
LDFLAGS+=-lm
CC=$(CXX)

ifeq (1,$(flx_have_lapack))
CXXFLAGS+=-DFLX_CONFIG_HAVE_LAPACK
LDFLAGS+=-llapack -lblas -lm
#CXXFLAGS+=-DFLX_CONFIG_HAVE_LAPACKE-I/usr/include/lapacke
#LDFLAGS+=-llapacke
endif

export CXXFLAGS
export LDFLAGS
export CC

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
