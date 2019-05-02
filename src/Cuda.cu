#include <stdio.h>

namespace gpu_cuda {

__device__ void cudaMakeArray(float *gpu_array, int N )
{
  cudaMalloc(&gpu_array, N*sizeof(float));
  cudaMemset(&gpu_array, 0, N*sizeof(float));
}

__device__ void cudaPutArray( float *gpu_array, float *cpu_array, int N )
{
  cudaMemcpy(gpu_array, cpu_array, N*sizeof(float), cudaMemcpyHostToDevice);
}

__device__ void cudaGetArray( float *cpu_array, float *gpu_array, int N )
{
  cudaMemcpy(cpu_array, gpu_array, N*sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace gpu
