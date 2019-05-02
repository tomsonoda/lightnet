#include <stdio.h>
#include "CudaObject.h"

namespace gpu_cuda {

__global__ void calcReluForwardGPU(float *in, float *out)
{
  // int i = blockIdx.x*blockDim.x + threadIdx.x;
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  float v = in[id];
  if ( v < 0 ){
    v = 0.0;
  }
  out[id] = v;
}

__global__ void calcReluBackwardGPU(float *dz_in, float *dz, float *in)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  dz[id] +=  (in[id] < 0) ? (0) : (1.0 * dz_in[id]);
}

void reluForwardGPU(float *in, float *out, int N)
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcReluForwardGPU<<<grid, BLOCK>>>(in, out);
}

void reluBackwardGPU( float *dz_in, float *dz, float *in, int N )
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcReluBackwardGPU<<<grid, BLOCK>>>( dz_in, dz, in );
}

} // namespace gpu
