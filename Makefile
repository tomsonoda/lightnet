COMPILER = g++
CFLAGS   = -g -Wall -O2
LDFLAGS  =
ALIB     = liblightnet.a
SLIB     = liblightnet.so
AR       = ar
ARFLAGS  = crs
COMMON   = -Iinclude/ -Isrc/
EXECOBJA = lightnet.o object_detection.o
TARGET   = lightnet
OBJDIR   = ./obj
SRCDIR   = ./src
SOURCES  = $(wildcard src/*.cpp)
OBJECTS  = $(addprefix $(OBJDIR)/,$(notdir $(SOURCES:.cpp=.o)))
VPATH    = ./src:./examples
DEPS     = $(wildcard src/*.hpp) Makefile include/lightnet.hpp
EXECOBJ  = $(addprefix $(OBJDIR)/, $(EXECOBJA))

all: $(TARGET) $(SLIB) $(ALIB)

$(TARGET): $(EXECOBJ) $(ALIB)
		$(COMPILER) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(OBJDIR)/%.o: %.cpp $(DEPS)
		mkdir -p obj
		$(COMPILER) $(CFLAGS) $(COMMON) -o $@ -c $<

$(ALIB): $(OBJECTS)
		$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJECTS)
		$(COMPILER) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

# $(warning SOURCES = $(SOURCES))

.PHONY: clean

clean:
		rm -f $(OBJECTS) $(TARGET) $(SLIB) $(ALIB) obj/*
