#include <stdio.h>
#include "CudaObject.h"
#include "CudaCommon.cuh"

namespace gpu_cuda {

__global__ void calcMaxPoolForwardGPU(
  float *in,float *out,
  int in_size_x, int in_size_y, int in_size_z,
  int batch_size, int out_size_x, int out_size_y, int out_size_z,
  int stride, int kernel_size
)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_out = id;

  if( id_out < batch_size * out_size_x * out_size_y * out_size_z) {
    int x = id % out_size_x;
    id /= out_size_x;
    int y = id % out_size_y;
    id /= out_size_y;
    int z = id % out_size_z;
    id /= out_size_z;
    int b = id;

    int mapped_x = x * stride;
    int mapped_y = y * stride;

    float mval = -1000000.0;
    for ( int j = 0; j < kernel_size; ++j ){
      for ( int i = 0; i < kernel_size; ++i ){

        int id_in = b * (in_size_z * in_size_x * in_size_y) +
          z * (in_size_x * in_size_y) +
          (mapped_y + j) * (in_size_x) +
          (mapped_x + i);

        float v = in[id_in];
        if ( v > mval ){
          mval = v;
        }
      }
    }
    out[id_out] = mval;
  }

  /* original
  for ( int b = 0; b < in.size.b; ++b ){
    for ( int z = 0; z < out.size.z; ++z ){
      for ( int y = 0; y < out.size.y; ++y ){
        for ( int x = 0; x < out.size.x; ++x ){
          TensorCoordinate mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
          float mval = -FLT_MAX;
          for ( int j = 0; j < kernel_size; ++j ){
            for ( int i = 0; i < kernel_size; ++i ){
              float v = in( b, mapped.x + i, mapped.y + j, z );
              if ( v > mval ){
                mval = v;
              }
            }
          }
          out( b, x, y, z ) = mval;
        }
      }
    }
  }

  */
}

__global__ void calcMaxPoolBackwardGPU( float *dz_in, float *dz, float *in, float *out, int batch_size, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int kernel_size, int stride )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if ( id < batch_size * dz_size_x * dz_size_y * dz_size_z ){
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

    // if(rn.min_y!=rn.max_y || rn.min_x!=rn.max_x){
    //   printf("#GPU (x,y)=(%d, %d), rn.min.y=%d, rn.max.y=%d, rn.min.x=%d, rn.max.x=%d\n", x, y, rn.min_y, rn.max_y, rn.min_x, rn.max_x);
    // }

    for ( int j = rn.min_y; j <= rn.max_y; ++j ){
      for ( int i = rn.min_x; i <= rn.max_x; ++i ){

        int out_index = b * (dz_in_size_x * dz_in_size_y * dz_in_size_z) +
          z * (dz_in_size_x * dz_in_size_y) +
          j * (dz_in_size_x) +
          i ;

        // if(in_value == out[out_index]  && i==3 && j==13){
        //   printf("#GPU in_value=%f, i=%d, j=%d, z=%d\n", in_value, i, j, z);
        // }
        int is_max = (in_value == out[out_index] ? 1 : 0);
        sum_error += is_max * dz_in[out_index];

      }
    }
    //   printf("#GPU sum_error=%f, x=%d, y=%d, z=%d\n", sum_error, x, y, z);
    dz[id_dz] += sum_error;
    // if(x==22 && z==5 && y==3){
    //   printf("#GPU dz=%f, sum_error=%f, x=%d, y=%d, z=%d\n", dz[id_dz], sum_error, x, y, z);
    // }

  }

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

void maxPoolForwardGPU(float *in, float *out, int in_size_x, int in_size_y, int in_size_z, int batch_size, int out_size_x, int out_size_y, int out_size_z, int kernel_size, int stride )
{
  int out_N = batch_size * out_size_x * out_size_y * out_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(out_N);
  calcMaxPoolForwardGPU<<<grid, BLOCK>>>(in, out, in_size_x, in_size_y, in_size_z, batch_size, out_size_x, out_size_y, out_size_z, stride, kernel_size);
}

void maxPoolBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, float *out, int batch_size, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int kernel_size, int stride )
{
  int in_N = batch_size * dz_in_size_x * dz_in_size_y * dz_in_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid_in = cuda.cudaGridSize(in_N);
  cudaAddFirstArrayToSecondArray<<<grid_in, BLOCK>>>( dz_next_layer, dz_in, in_N );

  int N = batch_size * dz_size_x * dz_size_y * dz_size_z;
  dim3 grid = cuda.cudaGridSize(N);
  calcMaxPoolBackwardGPU<<<grid, BLOCK>>>( dz_in, dz, in, out, batch_size, dz_size_x, dz_size_y, dz_size_z, dz_in_size_x, dz_in_size_y, dz_in_size_z, kernel_size, stride );
}

} // namespace gpu
