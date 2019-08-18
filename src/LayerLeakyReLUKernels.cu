#include <stdio.h>
#include "CudaObject.h"

namespace gpu_cuda {

__global__ void calcLeakyReluForwardGPU(float *in, float *out, int elements)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if( id < elements ){
    float v = in[id];
    if ( v < 0 ){
      v = 0.01;
    }
    out[id] = v;
  }

  /* original
  for( unsigned i = 0; i < data_size; ++i ){
    float v = in.data[i];
    if ( v < 0 ){
      v = 0.01;
    }
    out.data[i] = v;
  }
  */
}

__global__ void calcLeakyReluBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, int elements )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if( id < elements ){
    dz_in[id] += dz_next_layer[id];
    dz[id] += (in[id] < 0) ? (0.01) : (dz_in[id]);
  }

  /* original
  for( unsigned i = 0; i < data_size; ++i ){
    dz_in.data[i] += dz_next_layer.data[i];
    dz.data[i] +=  (in.data[i] < 0) ? (0.01) : (1.0 * dz_in.data[i]);
  }
  */
}

void leakyReluForwardGPU(float *in, float *out, int N)
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcLeakyReluForwardGPU<<<grid, BLOCK>>>(in, out, N);
}

void leakyReluBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, int N )
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcLeakyReluBackwardGPU<<<grid, BLOCK>>>( dz_next_layer, dz_in, dz, in, N );
}

} // namespace gpu
