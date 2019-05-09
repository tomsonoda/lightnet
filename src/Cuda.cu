#include <stdio.h>

#include "CudaObject.h"

namespace gpu_cuda {

__device__ unsigned int Rand(unsigned int randx)
{
  randx = randx*1103515245+12345;
  return randx&2147483647;
}

float *cudaMakeArray( int N )
{
  float *gpu_array;
  cudaMalloc((void **)&gpu_array, N*sizeof(float));
  cudaMemset(&gpu_array, 0, N*sizeof(float));
  return gpu_array;
}

void cudaCheckError(cudaError_t status)
{
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess){
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess){
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    }
}

void cudaPutArray( float *gpu_array, float *cpu_array, int N )
{
  cudaError_t status = cudaMemcpy(gpu_array, cpu_array, N*sizeof(float), cudaMemcpyHostToDevice);
  check_error(status);
}

void cudaGetArray( float *cpu_array, float *gpu_array, int N )
{
  cudaError_t status = cudaMemcpy(cpu_array, gpu_array, N*sizeof(float), cudaMemcpyDeviceToHost);
  check_error(status);
}

void cudaClearArray( float *gpu_array, int N )
{
  cudaMemset(&gpu_array, 0, N*sizeof(float));
}

__global__ void setRandom(float *gpu_array, int maxval )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  gpu_array[id] = 1.0f / maxval * Rand(id) / float( RAND_MAX );
}

void cudaMakeRandomArray(float *gpu_array, int N, int maxval )
{
  cudaMalloc(&gpu_array, N*sizeof(float));
  cudaMemset(&gpu_array, 0, N*sizeof(float));
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  setRandom<<<grid, BLOCK>>>( gpu_array, maxval );
}

} // namespace gpu
