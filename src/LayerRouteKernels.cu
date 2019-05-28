#include <stdio.h>
#include "CudaObject.h"
#include "CudaCommon.cuh"

namespace gpu_cuda {

__global__ void calcRouteForwardGPU(float *in, float *out, int in_size_x, int in_size_y, int in_size_z, int z_offset )
{
  // int i = blockIdx.x*blockDim.x + threadIdx.x;
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_in = id;

  int x = id % in_size_x;
  id /= in_size_x;
  int y = id % in_size_y;
  id /= in_size_y;
  int z = id % in_size_z;
  id /= in_size_z;
  int b = id;

  int id_out = b * (in_size_z * in_size_x * in_size_y) + (z + z_offset) * (in_size_x * in_size_y) + y * (in_size_x) + x;
  out[id_out] = in[id_in];

  /* original code
  for ( int b = 0; b < layer_in.size.b; ++b ){
    for ( int z = 0; z < layer_in.size.z; ++z ){
      for ( int y = 0; y < layer_in.size.y; y++ ){
        for ( int x = 0; x < layer_in.size.x; x++ ){
          out( b, x, y, z_offset+z ) = layer_in( b, x, y, z );
        }
      }
    }
  }
  */

}

__global__ void calcRouteBackwardGPU( float *dz_in, float *dz, int in_size_x, int in_size_y, int in_size_z, int z_offset )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_out = id;
  int x = id % in_size_x;
  id /= in_size_x;
  int y = id % in_size_y;
  id /= in_size_y;
  int z = id % in_size_z;
  id /= in_size_z;
  int b = id;

  int id_in = b * (in_size_z * in_size_x * in_size_y) + (z + z_offset) * (in_size_x * in_size_y) + y * (in_size_x) + x;
  dz[id_out] += dz_in[id_in];
  /*
  for ( int b = 0; b < layer_dz.size.b; ++b ){
    for ( int z = 0; z < layer_dz.size.z; ++z ){
      for ( int y = 0; y < layer_dz.size.y; y++ ){
        for ( int x = 0; x < layer_dz.size.x; x++ ){
          layer_dz( b, x, y, z ) += dz_in( b, x, y, z_offset+z );
        }
      }
    }
  }
  */
}

void routeForwardGPU(float *in, float *out, int N, int in_size_x, int in_size_y, int in_size_z, int z_offset )
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcRouteForwardGPU<<<grid, BLOCK>>>(in, out, in_size_x, in_size_y, in_size_z, z_offset );
}

void routeBackwardAddFirstArrayToSecondArrayGPU( float *dz_next_layer, float *dz_in, int N )
{
  CudaObject cuda = CudaObject();
  dim3 grid_in = cuda.cudaGridSize(N);
  cudaAddFirstArrayToSecondArray<<<grid_in, BLOCK>>>( dz_next_layer, dz_in, N );
}

void routeBackwardGPU(  float *dz_in, float *dz, int N, int in_size_x, int in_size_y, int in_size_z, int z_offset )
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcRouteBackwardGPU<<<grid, BLOCK>>>(  dz_in, dz, in_size_x, in_size_y, in_size_z, z_offset );
}

} // namespace gpu
