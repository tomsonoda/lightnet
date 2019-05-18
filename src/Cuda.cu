#include <stdio.h>

#include "CudaObject.h"

namespace gpu_cuda {

__device__ unsigned int Rand(unsigned int randx)
{
  randx = randx*1103515245+12345;
  return randx&2147483647;
}

__global__ void cudaFillArray(int N, float val, float *gpu_array)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) gpu_array[i] = val;
}

__global__ void setRandom(float *gpu_array, int maxval )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  gpu_array[id] = 1.0f / maxval * Rand(id) / float( RAND_MAX );
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

void cudaFillGpuArray(int N, float val, float * array)
{
  CudaObject cuda = CudaObject();
  dim3 grid_in = cuda.cudaGridSize(N);
  cudaFillArray<<<grid_in, BLOCK>>>( N, val, array );
  cudaCheckError(cudaPeekAtLastError());
}

float *cudaMakeArray( float *cpu_array, int N )
{
  float *gpu_array;
  size_t size = N * sizeof(float);
  cudaError_t status = cudaMalloc((void **)&gpu_array, size);
  cudaCheckError(status);

  if(cpu_array){
      cudaMemcpy(gpu_array, cpu_array, size, cudaMemcpyHostToDevice);
  } else {
      cudaFillGpuArray(N, 0, gpu_array, 1);
  }

  cudaMemset(&gpu_array, 0, N*sizeof(float));
  return gpu_array;
}

void cudaPutArray( float *gpu_array, float *cpu_array, int N )
{
  cudaError_t status = cudaMemcpy(gpu_array, cpu_array, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(status);
}

void cudaGetArray( float *cpu_array, float *gpu_array, int N )
{
  cudaError_t status = cudaMemcpy(cpu_array, gpu_array, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError(status);
}

void cudaClearArray( float *gpu_array, int N )
{
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

} // namespace gpu
