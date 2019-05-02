#include <stdio.h>
#include "CudaObject.h"

namespace gpu_cuda {

__global__ void calcLeakyReluForwardGPU(float *in, float *out)
{
  // int i = blockIdx.x*blockDim.x + threadIdx.x;
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  float v = in[id];
  if ( v < 0 ){
    v = 0.01;
  }
  out[id] = v;
}

__global__ void calcLeakyReluBackwardGPU( float *dz_in, float *dz, float *in)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  dz[id] +=  (in[id] < 0) ? (0.01) : (1.0 * dz_in[id]);
}

void leakyReluForwardGPU(float *in, float *out, int N)
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcLeakyReluForwardGPU<<<grid, BLOCK>>>(in, out);
}

void leakyReluBackwardGPU( float *dz_in, float *dz, float *in, int N )
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcLeakyReluBackwardGPU<<<grid, BLOCK>>>( dz_in, dz , in);
}

} // namespace gpu
