#include <stdio.h>

#ifdef GPU

namespace gpu_cuda {
  __global__ void cudaMakeArray(float *gpu_array, int N )
  {
    cudaMalloc(&gpu_array, N*sizeof(float));
  }

  __global__ void cudaPutArray( float *gpu_array, float *cpu_array, int N )
  {
    cudaMemcpy(gpu_array, cpu_array, N*sizeof(float), cudaMemcpyHostToDevice);
  }

  __global__ void cudaGetArray( float *cpu_array, float *gpu_array, int N )
  {
    cudaMemcpy(cpu_array, gpu_array, N*sizeof(float), cudaMemcpyDeviceToHost);
  }

} // namespace gpu

#endif
