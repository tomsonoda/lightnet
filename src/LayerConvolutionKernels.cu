#include <stdio.h>
#include "CudaObject.h"
#include "CudaCommon.cuh"

namespace gpu_cuda {

__global__ void calcConvolutionForwardPaddedInGPU( float *in, float *padded_in,
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

__global__ void calcConvolutionForwardGPU( float *out, float *padded_in, float *filters, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int out_size_x, int out_size_y, int out_size_z, int kernel_size, int stride, int filter_size)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_out = id;

  int x = id % out_size_x;
  id /= out_size_x;
  int y = id % out_size_y;
  id /= out_size_y;
  int filter = id % out_size_z;
  id /= out_size_z;
  int b = id;

  int mapped_x = x * stride;
  int mapped_y = y * stride;

  float sum = 0.0;
  for ( int z = 0; z < padded_in_size_z; ++z ){
    for ( int j = 0; j < kernel_size; ++j ){
      for ( int i = 0; i < kernel_size; ++i ){
        int filter_index = z * (kernel_size * kernel_size) + j * kernel_size + i;
        int padded_in_index = b * (padded_in_size_x * padded_in_size_y * padded_in_size_z) + z * (padded_in_size_x * padded_in_size_y) + (mapped_y + j) * (padded_in_size_x) + (mapped_x + i);
        sum += filters[filter * filter_size + filter_index] * padded_in[padded_in_index];
      }
    }
  }
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

__global__ void calcConvolutionBackwardGPU( float *dz_in, float *dz, float *filters, float *filter_grads, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int padding, int kernel_size, int stride, int filter_size )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_out = id;

  int x = id % dz_size_x;
  id /= dz_size_x;
  int y = id % dz_size_y;
  id /= dz_size_y;
  int z = id % dz_size_z;
  id /= dz_size_z;
  int b = id;


  if( x>=padding && y>=padding && x-padding<dz_size_x && y-padding<dz_size_y ){
    float sum_error = 0;

    range_t rn = map_to_output( x, y, dz_in_size_x, dz_in_size_y, kernel_size, stride );
    int filter_size = 1 * kernel_size * kernel_size * dz_size_z;

    for ( int i = rn.min_x; i <= rn.max_x; i++ ){
      int minx = i * stride;
      for ( int j = rn.min_y; j <= rn.max_y; j++ ){
        int miny = j * stride;
        int x_minx = x - minx;
        int y_miny = y - miny;

        for ( int k = rn.min_z; k <= rn.max_z; k++ ){
          int dz_in_index = b * (dz_in_size_z * dz_in_size_x * dz_in_size_y) + k * (dz_in_size_x * dz_in_size_y) + j * dz_in_size_x + i ;
          float d = dz_in[ dz_in_index ];
          int filter_index = k * filter_size + (z * (kernel_size * kernel_size) + y_miny * kernel_size + x_minx);
          sum_error += filters[filter_index] * d;

          int filter_grad_index = k * (filter_size * 2) + (z * (kernel_size * kernel_size) + y_miny * kernel_size + x_minx) + 0; // grad=0, grad_prev=1
          int padded_in_index = b * (padded_in_size_z * padded_in_size_x * padded_in_size_y) + z * (padded_in_size_x * padded_in_size_y) + y * padded_in_size_x + x;
          filter_grads[filter_grad_index] += padded_in[padded_in_index] * d;
        }
      }
    }

    int dz_index = b * (dz_size_z * dz_size_x * dz_size_y) + z * (dz_size_x * dz_size_y) + (y - padding) * dz_size_x + (x - padding);
    dz[dz_index] += sum_error;
  }

  /* original code
  for ( int b = 0; b < in.size.b; b++ ){
    for ( int x = 0; x < padded_in.size.x; x++ ){
      for ( int y = 0; y < padded_in.size.y; y++ ){
        tensor_range_t rn = map_to_output( x, y );
        for ( int z = 0; z < in.size.z; z++ ){

          float sum_error = 0;
          for ( int i = rn.min_x; i <= rn.max_x; i++ ){
            int minx = i * stride;
            for ( int j = rn.min_y; j <= rn.max_y; j++ ){
              int miny = j * stride;
              int x_minx = x - minx;
              int y_miny = y - miny;
              for ( int k = rn.min_z; k <= rn.max_z; k++ ){
                float d = dz_in( b, i, j, k );
                sum_error += filters[k].get( 0, x_minx, y_miny, z ) * d;
                filter_grads[k].get( 0, x_minx, y_miny, z ).grad += padded_in( b, x, y, z ) * d;
              }
            }
          }

          if(x>=padding && y>=padding && x-padding<in.size.x && y-padding<in.size.y ){
            dz( b, x-padding, y-padding, z ) += sum_error;
          }
        }

      }
    }
  }
  */
}

void convolutionForwardGPU( float *in, float *out, float *padded_in, float *filters, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int padding, int kernel_size, int stride, int filter_size )
{
  int in_size = batch_size * in_size_x * in_size_y * in_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid_in = cuda.cudaGridSize(in_size);
  calcConvolutionForwardPaddedInGPU<<<grid_in, BLOCK>>>(in, padded_in, in_size_x, in_size_y, in_size_z, padding);

  int out_size = batch_size * out_size_x * out_size_y * out_size_z;
  dim3 grid_out = cuda.cudaGridSize(out_size);
  calcConvolutionForwardGPU<<<grid_out, BLOCK>>>( out, padded_in, filters, padded_in_size_x, padded_in_size_y, padded_in_size_z, out_size_x, out_size_y, out_size_z, kernel_size, stride, filter_size);
}

void convolutionBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *padded_in, float *filters, float *filter_grads, int batch_size, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int padding, int kernel_size, int stride, int filter_size )
{
  CudaObject cuda = CudaObject();
  int in_N = batch_size * dz_in_size_x * dz_in_size_y * dz_in_size_z;
  dim3 grid_in = cuda.cudaGridSize(in_N);
  cudaAddFirstArrayToSecondArray<<<grid_in, BLOCK>>>( dz_next_layer, dz_in );

  int dz_N = batch_size * dz_size_x * dz_size_y * dz_size_z;
  dim3 grid_dz = cuda.cudaGridSize(dz_N);
  calcConvolutionBackwardGPU<<<grid_dz, BLOCK>>>( dz_in, dz, filters, filter_grads, dz_size_x, dz_size_y, dz_size_z, padded_in_size_x, padded_in_size_y, padded_in_size_z, padding, kernel_size, stride, filter_size );
}

} // namespace gpu
