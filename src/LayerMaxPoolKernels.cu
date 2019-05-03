#include <stdio.h>
#include "CudaObject.h"
#include "CudaCommon.cuh"

namespace gpu_cuda {

__global__ void calcMaxPoolForwardGPU(
  float *in,float *out,
  int in_size_x, int in_size_y, int in_size_z,
  int out_size_x, int out_size_y, int out_size_z,
  int stride, int kernel_size
)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_out = id;

  int x = id % out_size_x;
  id /= out_size_x;
  int y = id % out_size_y;
  id /= out_size_y;
  int z = id % out_size_z;
  id /= out_size_z;
  int b = id;

  int mapped_x = x * stride;
  int mapped_y = y * stride;

  float mval = -100000.0;
  for ( int j = 0; j < kernel_size; ++j ){
    for ( int i = 0; i < kernel_size; ++i ){

      int id_in = b * (in_size_z * in_size_x * in_size_y) +
        z * (in_size_x * in_size_y) +
        (mapped_x + i) * (in_size_x) +
        (mapped_y + j);

      float v = in[id_in];
      if ( v > mval ){
        mval = v;
      }
    }
  }
  out[id_out] = mval;
}

__global__ void calcMaxPoolBackwardGPU( float *dz_in, float *dz, float *in, float *out, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int kernel_size, int stride ){
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_dz = id;

  int x = id % dz_size_x;
  id /= dz_size_x;
  int y = id % dz_size_y;
  id /= dz_size_y;
  int z = id % dz_size_z;
  id /= dz_size_z;
  int b = id;

  range_t rn = map_to_output( x, y, dz_in_size_x, dz_in_size_y, kernel_size, stride );

  float sum_error = 0;
  float in_value = in[id_dz];
  for ( int j = rn.min_y; j <= rn.max_y; ++j ){
    for ( int i = rn.min_x; i <= rn.max_x; ++i ){

      int out_index = (b * (dz_in_size_x * dz_in_size_y * dz_in_size_z) +
        z * (dz_in_size_x * dz_in_size_y) +
        j * (dz_in_size_x) +
        i );

      int is_max = in_value == out[out_index] ? 1 : 0;
      sum_error += is_max * dz_in[out_index];
    }
  }
  dz[id_dz] += sum_error;
  /*
  for ( int b = 0; b < in.size.b; ++b ){
    for ( int y = 0; y < in.size.y; ++y ){
      for ( int x = 0; x < in.size.x; ++x ){
        range_t rn = map_to_output( x, y );
        for ( int z = 0; z < in.size.z; ++z ){
          float sum_error = 0;
          float in_value = in( b, x, y, z );
          for ( int j = rn.min_y; j <= rn.max_y; ++j ){
            for ( int i = rn.min_x; i <= rn.max_x; ++i ){
              int is_max = in_value == out( b, i, j, z ) ? 1 : 0;
              sum_error += is_max * dz_in( b, i, j, z );
            }
          }
          dz( b, x, y, z ) += sum_error;
        }
      }
    }
  }
  */
}

void maxPoolForwardGPU(float *in, float *out, int in_size_x, int in_size_y, int in_size_z, int out_size_b, int out_size_x, int out_size_y, int out_size_z, int kernel_size, int stride )
{
  int out_N = out_size_b * out_size_x * out_size_y * out_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(out_N);
  calcMaxPoolForwardGPU<<<grid, BLOCK>>>(in, out, in_size_x, in_size_y, in_size_z, out_size_x, out_size_y, out_size_z, stride, kernel_size);
}

void maxPoolBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, float *out, int batch_size, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int kernel_size, int stride )
{
  int in_N = batch_size * dz_in_size_x * dz_in_size_y * dz_in_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid_in = cuda.cudaGridSize(in_N);
  cudaAddFirstArrayToSecondArray<<<grid_in, BLOCK>>>( dz_next_layer, dz_in );

  int N = batch_size * dz_size_x * dz_size_y * dz_size_z;
  dim3 grid = cuda.cudaGridSize(N);
  calcMaxPoolBackwardGPU<<<grid, BLOCK>>>( dz_in, dz, in, out, dz_size_x, dz_size_y, dz_size_z, dz_in_size_x, dz_in_size_y, dz_in_size_z, kernel_size, stride );
}

} // namespace gpu
