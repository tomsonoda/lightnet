#include <stdio.h>
#include "CudaObject.h"
#include "CudaCommon.cuh"

namespace gpu_cuda {

__global__ void calcBatchNormalizationForwardGPU( float *in, float *out, float *mean, float *xmu, float *variance, float *inv_variance, float *xhat, float *gamma, float *beta, int batch_size, int in_size_x, int in_size_y, int in_size_z )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  float scale = 1.0f / (batch_size * in_size_x * in_size_y);
  float sum = 0;

  for ( int b = 0; b < batch_size; ++b ){
    for ( int j = 0; j < in_size_y; ++j ){
      for ( int i = 0; i < in_size_x; ++i ){
        int in_index = b * (in_size_x * in_size_y * in_size_z) + id * (in_size_x * in_size_y) + j * (in_size_x) + i;
        sum += in[in_index];
      }
    }
  }
  mean[id] = sum * scale;

  sum = 0;
  for ( int b = 0; b < batch_size; ++b ){
    for ( int j = 0; j < in_size_y; ++j ){
      for ( int i = 0; i < in_size_x; ++i ){
        int in_index = b * (in_size_x * in_size_y * in_size_z) + id * (in_size_x * in_size_y) + j * (in_size_x) + i;
        xmu[in_index] = in[in_index] - mean[id];
        sum += pow( xmu[in_index], 2 );
      }
    }
  }

  variance[id] = sum * scale;
  float invvar = 1.0f / sqrt(variance[id] + 0.00001f);
  float gmm = gamma[id];
  float bt = beta[id];
  inv_variance[id] = invvar;

  for ( int b = 0; b < batch_size; ++b ){
    for (int j = 0; j < in_size_y; ++j ){
      for (int i = 0; i < in_size_x; ++i ){
        int in_index = b * (in_size_x * in_size_y * in_size_z) + id * (in_size_x * in_size_y) + j * (in_size_x) + i;
        float v = xmu[in_index] * invvar;
        xhat[in_index] = v;
        out[in_index] = gmm * v + bt;
      }
    }
  }

  /* original code
  int filters = in_size_z;
  scale = 1.0f / (in_size_b * in_size_x * in_size_y);

  for ( int z = 0; z < filters; ++z ){

    float sum = 0;
    for ( int b = 0; b < in_size_b; ++b ){
      for ( int j = 0; j < in_size_y; ++j ){
        for ( int i = 0; i < in_size_x; ++i ){
            sum += in(b, i, j, z);
        }
      }
    }
    mean(0, 0, 0, z) = sum * scale;

    sum = 0;
    for ( int b = 0; b < in_size_b; ++b ){
      for ( int j = 0; j < in_size_y; ++j ){
        for ( int i = 0; i < in_size_x; ++i ){
          xmu( b, i, j, z ) = in(b, i, j, z) - mean(0, 0, 0, z);
          sum += pow( xmu( b, i, j, z ), 2 );
        }
      }
    }

    variance(0, 0, 0, z) = sum * scale;
    float invvar = 1.0f / sqrt(variance( 0, 0, 0, z )+0.00001f);
    float gmm = gamma( 0, 0, 0, z );
    float bt = beta( 0, 0, 0, z );
    inv_variance(0, 0, 0, z ) = invvar;

    for ( int b = 0; b < in_size_b; ++b ){
      for (int j = 0; j < in_size_y; ++j ){
        for (int i = 0; i < in_size_x; ++i ){
          float v = xmu( b, i, j, z ) * invvar;
          xhat( b, i, j, z ) = v;
          out( b, i, j, z ) = gmm * v + bt;
        }
      }
    }
  }
  */
}

__global__ void calcBatchNormalizationUpdateWeightsGPU( float *gamma, float *beta, float *dgamma, float *dbeta, float learning_rate )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  gamma[id] -= learning_rate * dgamma[id];
  beta[id]  -= learning_rate * dbeta[id];
  /*
  for( int i=0; i < in_size_z; ++i ){
    gamma.data[i] -= lr * dgamma.data[i];
    beta.data[i] -= lr * dbeta.data[i];
  }
  */
}

__global__ void calcBatchNormalizationBackwardGPU( float *dz_in, float *dz, float *xmu, float *variance, float *inv_variance, float *xhat, float *gamma, float *beta, float *dxhat, float *dx1, float *dgamma, float *dbeta, int batch_size, int in_size_x, int in_size_y, int in_size_z )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  float bxy_inv = 1.0f / (float)(batch_size * in_size_x * in_size_y);
  float dbeta_sum = 0.0;
  float dgamma_sum = 0.0;
  float dvariance_sum = 0.0;

  for ( int b = 0; b < batch_size; ++b ){
    for ( int j = 0; j < in_size_y; ++j ){
      for ( int i = 0; i < in_size_x; ++i ){
        int in_index = b * (in_size_x * in_size_y * in_size_z) + id * (in_size_x * in_size_y) + j * (in_size_x) + i;

        float delta = dz_in[in_index];
        dbeta_sum += delta;
        dgamma_sum += xhat[in_index] * delta;
        dvariance_sum += delta * xmu[in_index];
      }
    }
  }

  dbeta[id] = dbeta_sum;
  dgamma[id] = dgamma_sum;

  float divar = 0.0;
  float gmm = gamma[id];
  for ( int b = 0; b < batch_size; ++b ){
    for ( int j = 0; j < in_size_y; ++j ){
      for ( int i = 0; i < in_size_x; ++i ){
        int in_index = b * (in_size_x * in_size_y * in_size_z) + id * (in_size_x * in_size_y) + j * (in_size_x) + i;
        float v = dz_in[in_index] * gmm;
        dxhat[in_index] = v;
        divar += v * xmu[in_index];
      }
    }
  }

  float dmu = 0.0;
  float invvar = inv_variance[id];
  float invvar_sqrt2 = -1. /(variance[id]+0.00001f);

  for ( int b = 0; b < batch_size; ++b ){
    for ( int j = 0; j < in_size_y; ++j ){
      for ( int i = 0; i < in_size_x; ++i ){
        // float dxmu1 = dxhat( b, i, j, z ) * invvar;
        int in_index = b * (in_size_x * in_size_y * in_size_z) + id * (in_size_x * in_size_y) + j * (in_size_x) + i;
        float dxmu1 = dxhat[in_index];
        float dsqrtvar =  invvar_sqrt2 * divar;
        float dxmu2 = xmu[in_index] * bxy_inv * dsqrtvar;
        float sum_dxmu = (dxmu1 + dxmu2) * invvar;
        dx1[in_index] = sum_dxmu;
        dmu += -sum_dxmu;
      }
    }
  }

  float dx2 = dmu * bxy_inv;

  for ( int b = 0; b < batch_size; ++b ){
    for ( int j = 0; j < in_size_y; ++j ){
      for ( int i = 0; i < in_size_x; ++i ){
        int in_index = b * (in_size_x * in_size_y * in_size_z) + id * (in_size_x * in_size_y) + j * (in_size_x) + i;
        dz[in_index] =  dx1[in_index] + dx2;
      }
    }
  }




  /* // original code
  float bxy_inv = 1.0f / (float)(in_size_b * in_size_x * in_size_y);

  for ( int z = 0; z < in_size_z; ++z ){
    float dbeta_sum = 0.0;
    float dgamma_sum = 0.0;
    float dvariance_sum = 0.0;
    for ( int b = 0; b < in_size_b; ++b ){
      for ( int j = 0; j < in_size_y; ++j ){
        for ( int i = 0; i < in_size_x; ++i ){
          float delta = dz_in( b, i, j, z );
          dbeta_sum += delta;
          dgamma_sum += xhat( b, i, j, z ) * delta;
          dvariance_sum += delta * xmu( b, i, j, z );
        }
      }
    }

    dbeta( 0, 0, 0, z ) = dbeta_sum;
    dgamma( 0, 0, 0, z ) = dgamma_sum;

    float divar = 0.0;
    float gmm = gamma( 0, 0, 0, z );
    for ( int b = 0; b < in_size_b; ++b ){
      for ( int j = 0; j < in_size_y; ++j ){
        for ( int i = 0; i < in_size_x; ++i ){
          float v = dz_in( b, i, j, z ) * gmm;
          dxhat( b, i, j, z ) = v;
          divar += v * xmu( b, i, j, z );
        }
      }
    }

    float dmu = 0.0;
    float invvar = inv_variance( 0, 0, 0, z );
    float invvar_sqrt2 = -1. /(variance( 0, 0, 0, z )+0.00001f);

    for ( int b = 0; b < in_size_b; ++b ){
      for ( int j = 0; j < in_size_y; ++j ){
        for ( int i = 0; i < in_size_x; ++i ){
          // float dxmu1 = dxhat( b, i, j, z ) * invvar;
          float dxmu1 = dxhat( b, i, j, z );
          float dsqrtvar =  invvar_sqrt2 * divar;
          // float dvar = 0.5 * invvar * dsqrtvar;
          // float dxmu2 = 2 * xmu( b, i, j, z ) * bxy_inv * dvar;
          // float dxmu2 = xmu( b, i, j, z ) * bxy_inv * invvar * dsqrtvar;
          float dxmu2 = xmu( b, i, j, z ) * bxy_inv * dsqrtvar;
          float sum_dxmu = (dxmu1 + dxmu2) * invvar;
          dx1( b, i, j, z ) = sum_dxmu;
          dmu += -sum_dxmu;
        }
      }
    }

    float dx2 = dmu * bxy_inv;

    for ( int b = 0; b < in_size_b; ++b ){
      for ( int j = 0; j < in_size_y; ++j ){
        for ( int i = 0; i < in_size_x; ++i ){
          dz( b, i, j, z ) =  dx1( b, i, j, z ) + dx2;
        }
      }
    }

  }
  */
}

void batchNormalizationForwardGPU( float *in, float *out, float *mean, float *xmu, float *variance, float *inv_variance, float *xhat, float *gamma, float *beta, int batch_size, int in_size_x, int in_size_y, int in_size_z )
{
  int N = in_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcBatchNormalizationForwardGPU<<<grid, BLOCK>>>( in, out, mean, xmu, variance, inv_variance, xhat, gamma, beta, batch_size, in_size_x, in_size_y, in_size_z );
}

void batchNormalizationUpdateWeightsGPU( float *gamma, float *beta, float *dgamma, float *dbeta, float learning_rate, int in_size_z )
{
  int N = in_size_z;
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcBatchNormalizationUpdateWeightsGPU<<<grid, BLOCK>>>( gamma, beta, dgamma, dbeta, learning_rate );
}

void batchNormalizationBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *xmu, float *variance, float *inv_variance, float *xhat, float *gamma, float *beta, float *dxhat, float *dx1, float *dgamma, float *dbeta, int batch_size, int in_size_x, int in_size_y, int in_size_z )
{
  CudaObject cuda = CudaObject();
  int in_N = batch_size * in_size_x * in_size_y * in_size_z;
  dim3 grid_in = cuda.cudaGridSize(in_N);
  cudaAddFirstArrayToSecondArray<<<grid_in, BLOCK>>>( dz_next_layer, dz_in );

  int N = in_size_z;
  dim3 grid = cuda.cudaGridSize(N);
  calcBatchNormalizationBackwardGPU<<<grid, BLOCK>>>( dz_in, dz, xmu, variance, inv_variance, xhat, gamma, beta, dxhat, dx1, dgamma, dbeta, batch_size, in_size_x, in_size_y, in_size_z );
}

} // namespace gpu
