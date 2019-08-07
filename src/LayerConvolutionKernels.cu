#include <stdio.h>
#include "CudaObject.h"
#include "CudaCommon.cuh"

namespace gpu_cuda {

__global__ void calcConvolutionForwardPaddedInGPU( float *in, float *padded_in, int batch_size, int in_size_x, int in_size_y, int in_size_z, int padding)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if( id < batch_size * in_size_x * in_size_y * in_size_z ){
    int in_index = id;

    int x = id % in_size_x;
    id /= in_size_x;
    int y = id % in_size_y;
    id /= in_size_y;
    int z = id % in_size_z;
    id /= in_size_z;
    int b = id;

    int pad_index = b * (in_size_z * (in_size_x + 2*padding) * (in_size_y + 2*padding) ) +
    z * ((in_size_x + 2*padding) * (in_size_y + 2*padding)) +
    (y+padding) * (in_size_x + 2*padding) +
    (x+padding) ;

    padded_in[pad_index] = in[in_index];
  }
  /* original code
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

__global__ void calcConvolutionForwardGPU( float *out, float *padded_in, float *filters, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int batch_size, int out_size_x, int out_size_y, int out_size_z, int kernel_size, int stride, int filter_size)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_out = id;

  if (id_out < batch_size * out_size_x * out_size_y * out_size_z) {
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
    for ( int z = 0; z < padded_in_size_z; ++z ){ // padded_in_size_z = in_size_z
      for ( int j = 0; j < kernel_size; ++j ){
        for ( int i = 0; i < kernel_size; ++i ){

          int padded_in_index = b * (padded_in_size_x * padded_in_size_y * padded_in_size_z) + z * (padded_in_size_x * padded_in_size_y) + (mapped_y + j) * (padded_in_size_x) + (mapped_x + i);
          int filter_index = z * (kernel_size * kernel_size) + j * kernel_size + i;

          sum += filters[filter * filter_size + filter_index] * padded_in[padded_in_index];
        }
      }
    }
    out[id_out] = sum;
  }

  /* original code
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

__global__ void calcConvolutionUpdateWeightsGPU( float *filters, float *filter_grads, int in_size_z, int number_filters, int kernel_size, float momentum, float decay, float learning_rate, int elements )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if ( id < elements ) {

    int id_out = id;
    int i = id % kernel_size;
    id /= kernel_size;
    int j = id % kernel_size;
    id /= kernel_size;
    int z = id % in_size_z;
    id /= in_size_z;
    int filter = id;

    int filter_size = 1 * kernel_size * kernel_size * in_size_z;
    int filter_grad_index = (filter * filter_size + z * (kernel_size * kernel_size) + j * kernel_size + i) * 2;

    float grad = filter_grads[ filter_grad_index ];
    float grad_prev = filter_grads[ filter_grad_index + 1 ];
    float m = ( grad + grad_prev * momentum );

    filter_grads[ filter_grad_index + 1 ] = m;

    float w = filters[ id_out ];
    w -= learning_rate * ( m + (decay * w));
    filters[ id_out ] = w;
  }

  /* original code
  int filters_size = filters.size();
  for ( int a = 0; a < filters_size; ++a ){
    for ( int z = 0; z < in.size.z; ++z ){
      for ( int j = 0; j < kernel_size; ++j ){
        for ( int i = 0; i < kernel_size; ++i ){
          GradientObject& grad = filter_grads[a].get( 0, i, j, z );
          float m = (grad.grad + grad.grad_prev * momentum);
          grad.grad_prev = m;
          float& w = filters[a].get( 0, i, j, z );
          w -= lr * ( m + (decay * w));
        }
      }
    }
  }
  */
}

__global__ void calcConvolutionBackwardResetGradGPU( float *filter_grads, int in_size_z, int kernel_size, int filter_size, int elements )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if ( id < elements ) {
    int i = id % kernel_size;
    id /= kernel_size;
    int j = id % kernel_size;
    id /= kernel_size;
    int z = id % in_size_z;
    id /= in_size_z;
    int filter = id;

    int filter_grad_index = (filter * (in_size_z * kernel_size * kernel_size) + z * (kernel_size * kernel_size) + j * kernel_size + i) * 2;
    filter_grads[ filter_grad_index ] = 0;
  }

  /* original code
  int k_end = filter_grads.size();
  int kernel_size_2 = kernel_size * kernel_size;
  int i_end = kernel_size_2 * in.size.z;
  for ( int k = 0; k < k_end; ++k ){
    for ( int i = 0; i < i_end ; ++i ){
        filter_grads[k].data[i].grad = 0;
    }
  }
  */
}


__global__ void calcConvolutionBackwardFilterGradsGPU( float *dz_in, float *dz, float *padded_in, float *filters, float *filter_grads, int batch_size, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int padding, int kernel_size, int stride, int number_filters, int filter_size, int elements )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if ( id < elements ) {
    int i = id % kernel_size;
    id /= kernel_size;
    int j = id % kernel_size;
    id /= kernel_size;
    int z = id % padded_in_size_z;
    id /= padded_in_size_z;
    int k = id;

    for ( int b = 0; b < batch_size; ++b ){
      for ( int y = padding; y < (padded_in_size_y - padding); ++y ){
        for ( int x = padding; x < (padded_in_size_x - padding); ++x ){
          range_t rn = map_to_output( x, y, dz_in_size_x, dz_in_size_y, kernel_size, stride );

          // if( j>=(y-rn.max_y*stride) && j<=(y-rn.min_y*stride) ){
            int jj = ( y - j ) / stride;

            // if( i>=(x-rn.max_x*stride) && i<=(x-rn.min_x*stride) ){
              int ii = ( x - i ) / stride;

              int dz_in_index = b * (dz_in_size_z * dz_in_size_x * dz_in_size_y) + k * (dz_in_size_x * dz_in_size_y) + jj * dz_in_size_x + ii ;
              float d = dz_in[ dz_in_index ];

              int padded_in_index = b * (padded_in_size_z * padded_in_size_x * padded_in_size_y) + z * (padded_in_size_x * padded_in_size_y) + y * padded_in_size_x + x;
              int filter_index = k * (padded_in_size_z * kernel_size * kernel_size ) + (z * (kernel_size * kernel_size) + j * kernel_size + i);
              int filter_grad_index = filter_index << 1; // grad=0, grad_prev=1

              // filter_grads[k].get( 0, x_minx, y_miny, z ).grad += padded_in_value * d;
              filter_grads[filter_grad_index] += padded_in[padded_in_index] * d;
            // }

          // }

        }
      }
    }
  }

  /* original code
  int z_max = (int)filters.size();
  for ( int b = 0; b < in.size.b; ++b ){
    // code optimization.
    int dz_in_xy = dz_in.size.y * dz_in.size.x;
    int b_dz_in_xyz = b * dz_in.size.z * dz_in_xy;
    int padded_in_xy = padded_in.size.y * padded_in.size.x;
    int b_padded_in_xyz = b * padded_in.size.z * padded_in_xy;

    for ( int y = 0; y < (padded_in.size.y - padding); ++y ){
      for ( int x = 0; x < (padded_in.size.x - padding); ++x ){

        tensor_range_t rn = map_to_output( x, y );

        for ( int z = 0; z < in.size.z; ++z ){

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
                // filter_grads[k].get( 0, x_minx, y_miny, z ).grad += padded_in_value * d;
                filter_grads[k].data[xyz].grad += padded_in_value * d;
              }
            }

          }
        }
      }
    }
  }
  */

}


__global__ void calcConvolutionBackwardGPU( float *dz_in, float *dz, float *padded_in, float *filters, float *filter_grads, int batch_size, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int padding, int kernel_size, int stride, int number_filters, int filter_size )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  // int N = batch_size * padded_in_size_x * padded_in_size_y * dz_size_z;

  if ( id <  batch_size * padded_in_size_x * padded_in_size_y * dz_size_z ){

    int x = id % padded_in_size_x;
    id /= padded_in_size_x;
    int y = id % padded_in_size_y;
    id /= padded_in_size_y;
    int z = id % dz_size_z;
    id /= dz_size_z;
    int b = id;

    if( x>=padding && y>=padding && x<padded_in_size_x-padding && y<padded_in_size_y-padding ){

      range_t rn = map_to_output( x, y, dz_in_size_x, dz_in_size_y, kernel_size, stride );
      float sum_error = 0;

      for ( int j = rn.min_y; j <= rn.max_y; j++ ){
        int y_miny = y - j * stride;

        for ( int i = rn.min_x; i <= rn.max_x; i++ ){
          int x_minx = x - i * stride;

          for ( int k = 0; k <number_filters; k++ ){
            int dz_in_index = b * (dz_in_size_z * dz_in_size_x * dz_in_size_y) + k * (dz_in_size_x * dz_in_size_y) + j * dz_in_size_x + i ;
            float d = dz_in[ dz_in_index ];

            int filter_index = k * (padded_in_size_z * kernel_size * kernel_size ) + (z * (kernel_size * kernel_size) + y_miny * kernel_size + x_minx);
            sum_error += filters[filter_index] * d;
          }
        }
      }

      int dz_index = b * (dz_size_z * dz_size_x * dz_size_y) + z * (dz_size_x * dz_size_y) + (y - padding) * dz_size_x + (x - padding);
      dz[dz_index] += sum_error;
    }
  }

  /* original code
  int z_max = (int)filters.size();

  for ( int b = 0; b < in.size.b; ++b ){
    // code optimization.
    int dz_in_xy = dz_in.size.y * dz_in.size.x;
    int b_dz_in_xyz = b * dz_in.size.z * dz_in_xy;
    int padded_in_xy = padded_in.size.y * padded_in.size.x;
    int b_padded_in_xyz = b * padded_in.size.z * padded_in_xy;

    for ( int y = 0; y < (padded_in.size.y - padding); ++y ){
      for ( int x = 0; x < (padded_in.size.x - padding); ++x ){

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

  }
  */
}

void convolutionForwardGPU( float *in, float *out, float *padded_in, float *filters, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int padding, int kernel_size, int stride, int filter_size )
{
  int in_size = batch_size * in_size_x * in_size_y * in_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid_in = cuda.cudaGridSize(in_size);
  calcConvolutionForwardPaddedInGPU<<<grid_in, BLOCK>>>(in, padded_in, batch_size, in_size_x, in_size_y, in_size_z, padding);

  int out_size = batch_size * out_size_x * out_size_y * out_size_z;
  dim3 grid_out = cuda.cudaGridSize(out_size);
  calcConvolutionForwardGPU<<<grid_out, BLOCK>>>( out, padded_in, filters, padded_in_size_x, padded_in_size_y, padded_in_size_z, batch_size, out_size_x, out_size_y, out_size_z, kernel_size, stride, filter_size);
}

void convolutionUpdateWeightsGPU(float *filters, float *filter_grads, int in_size_z, int number_filters, int kernel_size, float momentum, float decay, float learning_rate)
{
  CudaObject cuda = CudaObject();
  int N = number_filters * kernel_size * kernel_size * in_size_z;
  dim3 grid = cuda.cudaGridSize(N);
  calcConvolutionUpdateWeightsGPU<<<grid, BLOCK>>>( filters, filter_grads, in_size_z, number_filters, kernel_size, momentum, decay, learning_rate, N );
}

void convolutionBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *padded_in, float *filters, float *filter_grads, int batch_size, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int padding, int kernel_size, int stride, int number_filters, int filter_size )
{
  CudaObject cuda = CudaObject();
  int in_N = batch_size * dz_in_size_x * dz_in_size_y * dz_in_size_z;
  dim3 grid_in = cuda.cudaGridSize(in_N);
  cudaAddFirstArrayToSecondArray<<<grid_in, BLOCK>>>( dz_next_layer, dz_in, in_N );

  int gN = number_filters * kernel_size * kernel_size * padded_in_size_z;
  dim3 grid = cuda.cudaGridSize(gN);
  calcConvolutionBackwardResetGradGPU<<<grid, BLOCK>>>( filter_grads, padded_in_size_z, kernel_size, filter_size, gN );

  calcConvolutionBackwardFilterGradsGPU<<<grid, BLOCK>>>( dz_in, dz, padded_in, filters, filter_grads, batch_size, dz_size_x, dz_size_y, dz_size_z, dz_in_size_x, dz_in_size_y, dz_in_size_z, padded_in_size_x, padded_in_size_y, padded_in_size_z, padding, kernel_size, stride, number_filters, filter_size, gN );

  // note: filter_size = 1 * kernel_size * kernel_size * in_size.z;
  // int gradN = number_filters * padded_in_size_x * padded_in_size_y * padded_in_size_z;
  // dim3 grid_grad = cuda.cudaGridSize(gradN);
  // calcConvolutionBackwardFilterGradsGPU<<<grid_grad, BLOCK>>>( dz_in, dz, padded_in, filters, filter_grads, batch_size, dz_size_x, dz_size_y, dz_size_z, dz_in_size_x, dz_in_size_y, dz_in_size_z, padded_in_size_x, padded_in_size_y, padded_in_size_z, padding, kernel_size, stride, number_filters, filter_size, gradN );

  int N = batch_size * padded_in_size_x * padded_in_size_y * dz_size_z;
  dim3 grid_dz = cuda.cudaGridSize(N);
  calcConvolutionBackwardGPU<<<grid_dz, BLOCK>>>( dz_in, dz, padded_in, filters, filter_grads, batch_size, dz_size_x, dz_size_y, dz_size_z, dz_in_size_x, dz_in_size_y, dz_in_size_z, padded_in_size_x, padded_in_size_y, padded_in_size_z, padding, kernel_size, stride, number_filters, filter_size );
}

} // namespace gpu
