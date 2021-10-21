# Lightnet #

Lightnet is a light deep neural network framework in C++. (Working...)

```
git clone https://github.com/tomsonoda/lightnet.git
```

Setup dataset.
```
(for MNIST)
chmod 755 scripts/setup_mnist_dataset.sh
bash scripts/setup_mnist_dataset.sh
mkdir checkpoints
bash scripts/run_mnist_conv2.sh
```

# For cuda programming

## Overview
https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf

## Device code note.

Device code format.
```
add<<< Dg, Db, Ns >>>();

Dg ... size of grid.
Db ... size of block.
Ns ... byte size of shared memory
```

Execute N times in parallel blocks.
```
add<<< N, 1 >>>();            while( atomicCAS(mutex, 0, 1) != 0 );
```
Execute N times in parallel threads.
```
add<<< 1, N >>>();
```

Device calculation...

```
* blockDim.x,y,z ... the number of threads in a block for each direction.
* gridDim.x,y,z ... the number of blocks in a grid for each direction.
* blockDim.x * gridDim.x ... the number of threads in a grid for the x direction.
```

NVIDIA document
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-variables

## CUDA shared memory limits.

 For compute capability 2.x and newer GPUs, they support up to 48kb of shared memory.
 For compute capability 1.x devices, the shared memory limit is 16kb (and up to 256 bytes of that can be consumed by kernel arguments).



 # References

http://yann.lecun.com/exdb/mnist/
