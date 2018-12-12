COMPILER = g++
CFLAGS   = -g -Wall -O3 -std=c++11
LDFLAGS  =
# ALIB     = liblightnet.a
# SLIB     = liblightnet.so
AR       = ar
ARFLAGS  = crs
COMMON   = -Iinclude/ -Isrc/
EXECOBJA = lightnet.o object_detection.o mnist.o
TARGET   = lightnet
OBJDIR   = ./obj
SRCDIR   = ./src
SOURCES  = $(wildcard src/*.cpp)
OBJECTS  = $(addprefix $(OBJDIR)/,$(notdir $(SOURCES:.cpp=.o)))
VPATH    = ./src:./examples
DEPS     = $(wildcard src/*.hpp) Makefile include/lightnet.hpp
EXECOBJ  = $(addprefix $(OBJDIR)/, $(EXECOBJA))

# all: $(TARGET) $(SLIB) $(ALIB)
all: $(TARGET)

# $(TARGET): $(EXECOBJ) $(ALIB)
# 		$(COMPILER) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(TARGET): $(EXECOBJ)
		$(COMPILER) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp $(DEPS)
		mkdir -p obj
		$(COMPILER) $(CFLAGS) $(COMMON) -o $@ -c $<

# $(ALIB): $(OBJECTS)
# 		$(AR) $(ARFLAGS) $@ $^

# $(SLIB): $(OBJECTS)
# 		$(COMPILER) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

# $(warning SOURCES = $(SOURCES))

.PHONY: clean

clean:
#   rm -f $(OBJECTS) $(TARGET) $(SLIB) $(ALIB) obj/*
		rm -f $(OBJECTS) $(TARGET) obj/* .DS_Store */.DS_Store
