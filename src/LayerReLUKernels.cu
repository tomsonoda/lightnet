#include <stdio.h>
#include "CudaObject.h"

namespace gpu_cuda {

__global__ void calcReluForwardGPU(float *in, float *out, int elements)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if( id < elements ){
    float v = in[id];
    if ( v < 0 ){
      v = 0.0;
    }
    out[id] = v;
  }

  /* original
  for( unsigned i = 0; i < data_size; ++i ){
    float v = in.data[i];
    if ( v < 0 ){
      v = 0;
    }
    out.data[i] = v;
  }
  */
}

__global__ void calcReluBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, int elements )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if( id < elements ){
    dz_in[id] += dz_next_layer[id];
    dz[id] += (in[id] < 0) ? (0) : (1.0 * dz_in[id]);
  }

  /* original
  for( unsigned i = 0; i < data_size; ++i ){
    dz_in.data[i] += dz_next_layer.data[i];
    dz.data[i] +=  (in.data[i] < 0) ? (0) : (1.0 * dz_in.data[i]);
  }
  */
}

void reluForwardGPU(float *in, float *out, int N)
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcReluForwardGPU<<<grid, BLOCK>>>(in, out, N);
}

void reluBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, int N )
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcReluBackwardGPU<<<grid, BLOCK>>>( dz_next_layer, dz_in, dz, in, N );
}

} // namespace gpu
