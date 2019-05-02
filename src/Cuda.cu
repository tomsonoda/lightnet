#include <stdio.h>

namespace gpu_cuda {

void cudaMakeArray(float *gpu_array, int N )
{
  cudaMalloc(&gpu_array, N*sizeof(float));
  cudaMemset(&gpu_array, 0, N*sizeof(float));
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

__global__ void cudaAddArray(float * dz_next_layer, float *dz_in)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  dz_in[id] += dz_next_layer[id];
}

} // namespace gpu
