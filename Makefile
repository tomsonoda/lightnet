COMPILER = gcc
CFLAGS   = -g -Wall -O2
LDFLAGS  =
ALIB     = liblightnet.a
SLIB     = liblightnet.so
AR       = ar
ARFLAGS  = rcs
COMMON   = -Iinclude/ -Isrc/
EXECOBJA = lightnet.o
TARGET   = lightnet
OBJDIR   = ./obj
SRCDIR   = ./src
SOURCES  = $(wildcard src/*.c)
OBJECTS  = $(addprefix $(OBJDIR)/,$(notdir $(SOURCES:.c=.o)))
VPATH    = ./src:./examples
DEPS     = $(wildcard src/*.h) Makefile include/lightnet.h
EXECOBJ  = $(addprefix $(OBJDIR)/, $(EXECOBJA))

all: $(TARGET) $(SLIB) $(ALIB)

$(TARGET): $(EXECOBJ) $(ALIB)
		$(COMPILER) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(OBJDIR)/%.o: %.c $(DEPS)
		mkdir -p obj
		$(COMPILER) $(CFLAGS) $(COMMON) -o $@ -c $<

$(ALIB): $(OBJECTS)
		$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJECTS)
		$(COMPILER) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

# $(warning SOURCES = $(SOURCES))


.PHONY: clean

clean:
		rm -f $(OBJECTS) $(TARGET) $(SLIB) $(ALIB)
