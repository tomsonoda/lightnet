# GPU_METAL=1
GPU_CUDA=1
ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]

NVCC= nvcc
COMPILER = g++
CFLAGS   = -g -Wall -O3 -std=c++11 -pthread
LDFLAGS  =
# ALIB     = liblightnet.a
# SLIB     = liblightnet.so
AR       = ar
ARFLAGS  = crs
COMMON   = -Iinclude/ -Isrc/
EXECOBJA = lightnet.o classification.o object_detection.o dataset.o
TARGET   = lightnet
OBJDIR   = ./obj
SRCDIR   = ./src
SOURCES  = $(wildcard src/*.cpp)
OBJECTS  = $(addprefix $(OBJDIR)/,$(notdir $(SOURCES:.cpp=.o)))
VPATH    = ./src:./examples
DEPS     = $(wildcard src/*.h) $(wildcard src/*.hpp) Makefile include/lightnet.h

ifeq ($(GPU_CUDA), 1)
COMMON+= -DGPU_CUDA -I/usr/local/cuda/include/
CFLAGS+= -DGPU_CUDA
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lstdc++
EXECOBJA+= Cuda.o LayerConvolutionKernels.o LayerLeakyReLUKernels.o LayerMaxPoolKernels.o
endif

EXECOBJ  = $(addprefix $(OBJDIR)/, $(EXECOBJA))

# GPU_METAL
# ifeq ($(GPU_METAL), 1)
# EXECOBJA = lightnet.o object_detection.o dataset.o
# COMPILER = clang++
# CFLAGS += -mmacosx-version-min=10.14 -DGPU_METAL
# LDFLAGS  = -framework Metal -framework MetalKit -framework Cocoa -framework CoreFoundation -fobjc-link-runtime
# METAL_OBJ_FLAGS = -std=c++11 -x objective-c++ -mmacosx-version-min=10.12 -Wdeprecated-declarations
# EXECOBJA += mtlpp.o
# GPU_METAL_SOURCE  = gpu_metal.metal
# TARGET += gpu_metal.metallib
# endif

# all: $(TARGET) $(SLIB) $(ALIB)
all: $(TARGET) obj

# $(TARGET): $(EXECOBJ) $(ALIB)
# 		$(COMPILER) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(TARGET): $(EXECOBJ)
		$(COMPILER) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp $(DEPS)
		$(COMPILER) $(CFLAGS) $(COMMON) -o $@ -c $<

$(OBJDIR)/%.o: %.cu $(DEPS)
		$(NVCC) -c $< -o $@


ifeq ($(GPU_METAL), 1)
$(OBJDIR)/mtlpp.o: mtlpp.mm $(DEPS)
		$(COMPILER) $(METAL_OBJ_FLAGS) -c $(SRCDIR)/mtlpp.mm -o $(OBJDIR)/mtlpp.o

gpu_metal.metallib: $(OBJDIR)/gpu_metal.air
		xcrun -sdk macosx metallib $(OBJDIR)/gpu_metal.air -o gpu_metal.metallib

$(OBJDIR)/gpu_metal.air: $(SRCDIR)/$(GPU_METAL_SOURCE)
		xcrun -sdk macosx metal -c $(SRCDIR)/$(GPU_METAL_SOURCE) -o $(OBJDIR)/gpu_metal.air
endif

# $(ALIB): $(OBJECTS)
# 		$(AR) $(ARFLAGS) $@ $^

# $(SLIB): $(OBJECTS)
# 		$(COMPILER) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

# $(warning SOURCES = $(SOURCES))

.PHONY: clean

obj:
	mkdir -p obj

clean:
#   rm -f $(OBJECTS) $(TARGET) $(SLIB) $(ALIB) obj/*
		rm -f $(OBJECTS) $(TARGET) obj/* .DS_Store */.DS_Store
