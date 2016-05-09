#inc_path=../include
#src_path=../src
#CXXFLAGS+=-Wall -Wextra -ansi -pedantic
#CXXFLAGS+=-g -Og
#CXXFLAGS+=-I$(inc_path)
#CXXFLAGS+=-I$(HOME)/sys/src/git/boost
#CXXFLAGS+=-I$(HOME)/sys/src/git/fuzzylite/fuzzylite
#LDFLAGS+=-L$(HOME)/sys/src/git/boost/stage/libs
#LDFLAGS+=-L$(HOME)/sys/src/git/fuzzylite/fuzzylite/debug/bin -lfuzzylited
#LDFLAGS+=-lm
#CC=$(CXX)

#components := anfis cluster detail
#srcdir := ./src
#bindir := ./build
#builddir := $(bindir)/.build
#srcdirs := $(srcdir) $(addprefix $(srcdir)/,$(components))
#sources := $(wildcard $(addsuffix /*.cpp,$(srcdirs)))
#objs := $(patsubst $(srcdir)/%,$(builddir)/%,$(patsubst %.cpp,%.o,$(sources)))

.PHONY: all clean

all: static shared

static:
#static: static_msg static_build

static_msg:
	@echo "====== Building Static Library ======"

static_build: $(objs)
#	ar rcs $(bindir)/libfuzzylitex-static.a $^
#	rm -rf $(objs)

shared: CXXFLAGS := $(CXXFLAGS) -fPIC
shared: shared_msg shared_build

shared_msg:
	@echo "====== Building Dynamic Library ======"

shared_build: $(objs)
	$(CXX) -o $(bindir)/libfuzzylitex.so.1.1.1 -shared -Wl,-soname,libfuzzylitex.so.1 $^
	ln -sf $(bindir)/libfuzzylitex.so.1.1.1 $(bindir)/libfuzzylitex.so.1
	ln -sf $(bindir)/libfuzzylitex.so.1.1.1 $(bindir)/libfuzzylitex.so
	rm -rf $(objs)

#$(bindir)/% : $(builddir)/%.o $(headers)
#	@echo "=== Building Target: $@ ==="
#	$(CXX) -o $@ $<  $(buildtmpdir)/boost/iostreams/src/file_descriptor.$(obj_ext) $(LDFLAGS) 

$(builddir)/%.o: $(srcdir)/%.cpp
	@echo "=== Compiling: $@ ==="
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

#engine.o: $(src_path)/anfis/engine.cpp
#	$(CXX) $(CXXFLAGS) -c -o engine.o $(src_path)/anfis/engine.cpp
#
#nodes.o: $(src_path)/anfis/nodes.cpp
#	$(CXX) $(CXXFLAGS) -c -o nodes.o $(src_path)/anfis/nodes.cpp
#
#subtractive.o: $(src_path)/cluster/subtractive.cpp
#	$(CXX) $(CXXFLAGS) -c -o subtractive.o $(src_path)/cluster/subtractive.cpp
#
#terms.o: $(src_path)/detail/terms.cpp
#	$(CXX) $(CXXFLAGS) -c -o terms.o $(src_path)/detail/terms.cpp
#
#training_gradient_descent.o: $(src_path)/anfis/training/gradient_descent.cpp
#	$(CXX) $(CXXFLAGS) -c -o training_gradient_descent.o $(src_path)/anfis/training/gradient_descent.cpp
#
#training_jang1993_hybrid.o: $(src_path)/anfis/training/jang1993_hybrid.cpp
#	$(CXX) $(CXXFLAGS) -c -o training_jang1993_hybrid.o $(src_path)/anfis/training/jang1993_hybrid.cpp
#
#training_training_algorithm.o: $(src_path)/anfis/training/training_algorithm.cpp
#	$(CXX) $(CXXFLAGS) -c -o training_training_algorithm.o $(src_path)/anfis/training/training_algorithm.cpp

clean:
	#rm -rf $(builddir)/*
	rm -rf $(objs)
	rm -f libfuzzylitex-static.a
	rm -f libfuzzylitex.so
	rm -f libfuzzylitex.so.*
