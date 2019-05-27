#include <stdio.h>

#include "CudaObject.h"

namespace gpu_cuda {

__device__ unsigned int Rand(unsigned int randx)
{
  randx = randx*1103515245+12345;
  return randx&2147483647;
}

__global__ void cudaFillArray( float *gpu_array, float val, int N )
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if( i < N ){
      gpu_array[i] = val;
    }
}

__global__ void setRandom(float *gpu_array, int N, int maxval )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if( id < N ){
    gpu_array[id] = 1.0f / maxval * Rand(id) / float( RAND_MAX );
  }
}

void cudaCheckError(cudaError_t status)
{
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess){
        const char *s = cudaGetErrorString(status);
        printf("CUDA Error: %s\n", s);
    }
    if (status2 != cudaSuccess){
        const char *s = cudaGetErrorString(status);
        printf("CUDA Error Prev: %s\n", s);
    }
}

void cudaFillGpuArray( float *array, float val, int N )
{
  CudaObject cuda = CudaObject();
  dim3 grid_in = cuda.cudaGridSize(N);
  cudaFillArray<<<grid_in, BLOCK>>>( array, val, N );
  cudaCheckError(cudaPeekAtLastError());
}

float *cudaMakeArray( float *cpu_array, int N )
{
  float *gpu_array;
  size_t size = N * sizeof(float);
  cudaError_t status = cudaMalloc((void **)&gpu_array, size);
  cudaCheckError(status);

  if(cpu_array){
      cudaMemcpy( gpu_array, cpu_array, size, cudaMemcpyHostToDevice );
  } else {
      cudaFillGpuArray( gpu_array,  0, N );
  }

  return gpu_array;
}

void cudaPutArray( float *gpu_array, float *cpu_array, size_t N )
{
  size_t size = N * sizeof(float);
  cudaError_t status = cudaMemcpy(gpu_array, cpu_array, size, cudaMemcpyHostToDevice);
  cudaCheckError(status);
}

void cudaGetArray( float *cpu_array, float *gpu_array, size_t N )
{
  size_t size = N * sizeof(float);
  cudaError_t status = cudaMemcpy(cpu_array, gpu_array, size, cudaMemcpyDeviceToHost);
  cudaCheckError(status);
}

void cudaClearArray( float *gpu_array, int N )
{
  // cudaFillGpuArray( gpu_array, 0, N);
  // cudaMemset(&gpu_array, 0, N*sizeof(float));
}

float *cudaMakeRandomArray(int N, int maxval )
{
  float *gpu_array = cudaMakeArray( NULL, N );
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  setRandom<<<grid, BLOCK>>>( gpu_array, N, maxval );
  return gpu_array;
}

} // namespace gpu
