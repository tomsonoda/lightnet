#include <stdio.h>
#include "CudaObject.h"

namespace gpu_cuda {

__global__ void calcConvolutionForwardPaddedInGPU(float *in, float *padded_in,
    int in_size_x, int in_size_y, int in_size_z, int padding)
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

  if( (x - padding ) > 0 && ( y - padding ) > 0){
    int in_index = (b * (in_size_z * in_size_x * in_size_y) +
    z * (in_size_x * in_size_y) +
    (y - padding) * (in_size_x) +
    (x - padding) );
    padded_in[id_out] = in[in_index];
  }
  /*
  for ( int b = 0; b < in.size.b; ++b ){
    for ( int z = 0; z < in.size.z; ++z ){
      for ( int y = 0; y < in.size.y; ++y ){
        for ( int x = 0; x < in.size.x; ++x ){
          padded_in( b, padding+x, padding+y, z ) = in( b, x, y, z );
        }
      }
    }
  }
  */
}

__global__ void calcConvolutionForwardGPU( float *padded_in, float *out,
    int out_size_x, int out_size_y, int out_size_z)
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

  int out_index = b * (out_size_z * out_size_x * out_size_y) +
  z * (out_size_x * out_size_y) +
  y * (out_size_x) +
  x ;
  float sum = 0.0;

  out[id_out] = sum;
  /*
  for ( int b = 0; b < in.size.b; ++b ){
    int filters_size = filters.size();
    for ( int filter = 0; filter < filters_size; ++filter ){
      TensorObject<float> filter_data = filters[filter];
      for ( int y = 0; y < out.size.y; ++y ){
        for ( int x = 0; x < out.size.x; ++x ){
          TensorCoordinate mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
          float sum = 0;

          for ( int z = 0; z < in.size.z; ++z ){
            for ( int j = 0; j < kernel_size; ++j ){
              for ( int i = 0; i < kernel_size; ++i ){
                sum += filter_data( 0, i, j, z ) * padded_in( b, mapped.x + i, mapped.y + j, z );
              }
            }
          }
          out( b, x, y, filter ) = sum;
        }
      }
    }
  }*/
}

__global__ void calcConvolutionBackwardGPU(float *in1, float *in2, float *in3, float* out)
{

}

void convolutionForwardGPU(float *in, float *out, float *padded_in,
  int batch_size,
  int in_size_x, int in_size_y, int in_size_z,
  int out_size_x, int out_size_y, int out_size_z,
  int padding)
{

  int in_size = batch_size * in_size_x * in_size_y * in_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid_in = cuda.cudaGridSize(in_size);
  calcConvolutionForwardPaddedInGPU<<<grid_in, BLOCK>>>(in, padded_in, in_size_x, in_size_y, in_size_z, padding);

  int out_size = batch_size * out_size_x * out_size_y * out_size_z;
  dim3 grid_out = cuda.cudaGridSize(out_size);
  calcConvolutionForwardGPU<<<grid_out, BLOCK>>>(padded_in, out, out_size_x, out_size_y, out_size_z);
}

void convolutionBockwardGPU(float *dz_next_layer, float *dz_in, float *dz, int N)
{
}

} // namespace gpu
