#include <stdio.h>
#include "CudaObject.h"
#include "CudaCommon.cuh"

namespace gpu_cuda {


__global__ void calcDenseForwardGPU( float *in, float *out, float *weights, float *biases, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_out = id;


  if ( id_out < batch_size * out_size_x * out_size_y * out_size_z ){
    int n = id % out_size_x;
    id /= out_size_x;
    // int y = id % out_size_y;
    id /= out_size_y;
    // int z = id % out_size_z;
    id /= out_size_z;
    int b = id;

    int w_size_x = in_size_x*in_size_y*in_size_z;

    float sum = 0;
    for ( int k = 0; k < in_size_z; ++k ){
      for ( int j = 0; j < in_size_y; ++j ){
        for ( int i = 0; i < in_size_x; ++i ){
          int m = k * (in_size_x * in_size_y) + j * (in_size_x) + i;
          int w_index = n * (w_size_x) + m;
          int in_index = b * (in_size_x * in_size_y * in_size_z) + k * (in_size_x * in_size_y) + j * in_size_x + i;
          sum += in[in_index] * weights[w_index];
        }
      }
    }
    int bias_index = n;
    out[id_out] = sum + biases[bias_index];
  }

  /*
  for ( int b = 0; b < in.size.b; ++b ){
    for ( int n = 0; n < out.size.x; ++n ){
      float sum = 0;
      for ( int z = 0; z < in.size.z; ++z ){
        for ( int j = 0; j < in.size.y; ++j ){
          for ( int i = 0; i < in.size.x; ++i ){
            int m = map( { 0, i, j, z } );
            sum += in( b, i, j, z ) * weights( 0, m, n, 0 );
          }
        }
      }
      out( b, n, 0, 0 ) = sum + biases( 0, 0, n, 0);
    }
  }
  */
}

__global__ void calcDenseUpdateWeightsGPU( float *weights, float *biases, float *gradients, float *dW, float *dB, float learning_rate, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, int momentum )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int w_size_x = in_size_x*in_size_y*in_size_z;
  int w_size_y = out_size_x;

  for(int h=0; h<w_size_x; ++h){
      int index = id * (w_size_x * w_size_y) + h;
      weights[index] = weights[index] - learning_rate * 	dW[index];
  }

  biases[id] = biases[id] - learning_rate * dB[id];

  for( int b=0; b<batch_size; b++ ){
    int index = (b * out_size_x + id) * 2;
    gradients[index+1] = gradients[index] + gradients[index+1] * momentum;
  }
  /*
  for (int i=0; i<weigts_data_num; ++i){
    weights.data[i] = weights.data[i] - lr * 	dW.data[i];
  }

  for (int i=0; i<out.size.x; ++i){
    biases.data[i] = biases.data[i] - lr * 	dB.data[i];
  }

  for ( int i = 0; i < out.size.x * in.size.b; ++i ){
      GradientObject& grad = gradients[ i ];
      grad.grad_prev = (grad.grad + grad.grad_prev * _momentum);
  }
  */
}

__global__ void calcDenseBackwardGPU( float *dz_in, float *dz, float *in, float *weights, float *biases, float *gradients, float *dW, float *dB, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, float momentum, float decay )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_in = id;

  int i = id % in_size_x;
  id /= in_size_x;
  int j = id % in_size_y;
  id /= in_size_y;
  int z = id % in_size_z;
  id /= in_size_z;
  int b = id;

  int w_size_x = in_size_x*in_size_y*in_size_z;
  int w_size_y = out_size_x;

  int m = z * (in_size_x * in_size_y) + j * (in_size_x) + i;

  for ( int n = 0; n < out_size_x; ++n ){
    float dzin = dz_in[b * (in_size_x * in_size_y * in_size_z) + n];
    gradients[ n*batch_size + b ] = dzin;

    int w_index = n * (w_size_x * w_size_y) + m;
    float w = weights[w_index];

    dW[w_index] += in[id_in] * (gradients[ (n*batch_size + b) * 2 ] + gradients[ (n*batch_size + b) * 2 + 1 ] * momentum) + (decay * w);
    dz[id_in] += dzin * w;
  }

  /*
  for ( int n = 0; n < out.size.x; ++n ){
      for ( int z = 0; z < in.size.z; ++z ){
        for ( int j = 0; j < in.size.y; ++j ){
          for ( int i = 0; i < in.size.x; ++i ){
            int m = map( { 0, i, j, z } );

            for( int b = 0; b < in.size.b; ++b ){
              GradientObject& grad = gradients[ n*in.size.b + b ];
              float dzin = dz_in( b, n, 0, 0 );
              float w = weights(0, m, n, 0);
              float bias = biases( 0, 0, n, 0 );
              grad.grad = dzin;
              dW( 0, m, n, 0 ) += in( b, i, j, z ) * (grad.grad + grad.grad_prev * _momentum) + (_decay * w);
              dz( b, i, j, z ) += dzin * w;
            }
          }
        }
      }
  }
  */
}

__global__ void calcDenseBarckwardNabraBGPU( float *dz_in, float *dB, int batch_size, int out_size_x ){
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  for( int b = 0; b < batch_size; ++b ){
    dB[id] += dz_in[ b * (out_size_x) + id ];
  }
  /* original
  for ( int n = 0; n < out.size.x; ++n ){
    for( int b = 0; b < in.size.b; ++b ){
      dB( 0, 0, n, 0 ) += dz_in( b, n, 0, 0 );
    }
  }
  */
}

void denseForwardGPU( float *in, float *out, float *weights, float *biases, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z )
{
  int N = batch_size * out_size_x * out_size_y * out_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcDenseForwardGPU<<<grid, BLOCK>>>(in, out, weights, biases, batch_size, in_size_x, in_size_y, in_size_z, out_size_x, out_size_y, out_size_z );
}

void denseUpdateWeightsGPU( float *weights, float *biases, float *gradients, float *dW, float *dB, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, float learning_rate, int momentum )
{
  int N = out_size_x;
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcDenseUpdateWeightsGPU<<<grid, BLOCK>>>( weights, biases, gradients, dW, dB, batch_size, in_size_x, in_size_y, in_size_z, out_size_x, out_size_y, out_size_z, learning_rate, momentum );
}

void denseBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, float *weights, float *biases, float *gradients, float *dW, float *dB, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, float momentum, float decay  )
{
  int out_N = batch_size * out_size_x * out_size_y * out_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid_in = cuda.cudaGridSize(out_N);
  cudaAddFirstArrayToSecondArray<<<grid_in, BLOCK>>>( dz_next_layer, dz_in );

  int in_N = batch_size * in_size_x * in_size_y * in_size_z;
  dim3 grid = cuda.cudaGridSize(in_N);
  calcDenseBackwardGPU<<<grid, BLOCK>>>( dz_in, dz, in, weights, biases, gradients, dW, dB, batch_size, in_size_x, in_size_y, in_size_z, out_size_x, out_size_y, out_size_z, momentum, decay );

  int in_B = out_size_x;
  dim3 grid_B = cuda.cudaGridSize(in_B);
  calcDenseBarckwardNabraBGPU<<<grid_B, BLOCK>>>( dz_in, dB, batch_size, out_size_x );

}

} // namespace gpu
