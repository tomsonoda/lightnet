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
/*
int k_end = filter_grads.size();
int kernel_size_2 = kernel_size * kernel_size;
int i_end = kernel_size_2 * in.size.z;
for ( int k = 0; k < k_end; ++k ){
  for ( int i = 0; i < i_end ; ++i ){
      filter_grads[k].data[i].grad = 0;
  }
}

int z_max = (int)filters.size();
std::vector< std::future<int> > thread_results;

for ( int b = 0; b < in.size.b; ++b ){

  thread_results.emplace_back(thread_pool.enqueue([&, b] {

    // code optimization.
    int dz_in_xy = dz_in.size.y * dz_in.size.x;
    int b_dz_in_xyz = b * dz_in.size.z * dz_in_xy;
    int padded_in_xy = padded_in.size.y * padded_in.size.x;
    int b_padded_in_xyz = b * padded_in.size.z * padded_in_xy;

    int y_end = padded_in.size.y - padding;
    for ( int y = 0; y < y_end; ++y ){

      int x_end = padded_in.size.x - padding;
      for ( int x = 0; x < x_end; ++x ){

        tensor_range_t rn = map_to_output( x, y );

        for ( int z = 0; z < in.size.z; ++z ){

          float sum = 0;
          // float padded_in_value = padded_in( b, x, y, z );
          float padded_in_value = padded_in.data[( b_padded_in_xyz ) + (z * padded_in_xy) + (y * padded_in.size.x) + x];

          for ( int j = rn.min_y; j <= rn.max_y; ++j ){
            int y_miny = y - j * stride;

            for ( int i = rn.min_x; i <= rn.max_x; ++i ){
              int x_minx = x - i * stride;

              int xyz = z * kernel_size_2 + y_miny * kernel_size + x_minx; // ( 0, x_minx, y_miny, z )

              for ( int k = 0; k < z_max; ++k ){
                // float d = dz_in( b, i, j, k );
                float d = dz_in.data[ b_dz_in_xyz + (k * dz_in_xy) + (j * dz_in.size.x) + i];
                // sum += filters[k].get( 0, x_minx, y_miny, z ) * d;
                sum += filters[k].data[xyz] * d;
                // filter_grads[k].get( 0, x_minx, y_miny, z ).grad += padded_in_value * d;
                filter_grads[k].data[xyz].grad += padded_in_value * d;
              }
            }
          }

          if( x>=padding && y>=padding ){
            dz( b, x - padding, y - padding, z ) += sum;
          }
        }

      }
    }
*/
}

void convolutionForwardGPU( float *in, float *out, float *padded_in, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, int padding )
{
  int in_size = batch_size * in_size_x * in_size_y * in_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid_in = cuda.cudaGridSize(in_size);
  calcConvolutionForwardPaddedInGPU<<<grid_in, BLOCK>>>(in, padded_in, in_size_x, in_size_y, in_size_z, padding);

  int out_size = batch_size * out_size_x * out_size_y * out_size_z;
  dim3 grid_out = cuda.cudaGridSize(out_size);
  calcConvolutionForwardGPU<<<grid_out, BLOCK>>>(padded_in, out, out_size_x, out_size_y, out_size_z);
}

void convolutionBockwardGPU( float *dz_in, float *dz, float *padded_in )
{

}

} // namespace gpu
