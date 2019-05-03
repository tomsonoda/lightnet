#include <stdio.h>
#include <stdlib.h>

#include "CudaObject.h"

namespace gpu_cuda {

__global__ void setRandom(float *gpu_array, int maxval )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  gpu_array[id] = 1.0f / maxval * rand() / float( RAND_MAX );
}

void cudaMakeArray(float *gpu_array, int N )
{
  cudaMalloc(&gpu_array, N*sizeof(float));
  cudaMemset(&gpu_array, 0, N*sizeof(float));
}

void cudaMakeRandomArray(float *gpu_array, int N, int maxval )
{
  cudaMalloc(&gpu_array, N*sizeof(float));
  cudaMemset(&gpu_array, 0, N*sizeof(float));
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  setRandom<<<grid, BLOCK>>>( gpu_array, maxval );
}

void cudaPutArray( float *gpu_array, float *cpu_array, int N )
{
  cudaMemcpy(gpu_array, cpu_array, N*sizeof(float), cudaMemcpyHostToDevice);
}

void cudaGetArray( float *cpu_array, float *gpu_array, int N )
{
  cudaMemcpy(cpu_array, gpu_array, N*sizeof(float), cudaMemcpyDeviceToHost);
}

void cudaClearArray( float *gpu_array, int N ){
  cudaMemset(&gpu_array, 0, N*sizeof(float));
}

} // namespace gpu
