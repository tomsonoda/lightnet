#include <stdio.h>
#include "Cuda.hpp"
#include "TensorObject.h"
#include "TensorCoordinate.h"

namespace gpu_cuda {

__global__ void calcMaxPoolForwardGPU(float *in,float *out,
  int in_size_x, int in_size_y, int in_size_z,
  int size_x, int size_y, int size_z,
  int stride, int kernel_size)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;


  int x = id % size_x;
  int y = ((id - x) / size_x) % size_y
  int z = ((id - x - (y*size_x)) / (size_x * size_y)) % size_z;
  int b = (id - x - (y*size_x) - (z*size_x*size_y)) / (size_z * size_y * size_x);

  TensorCoordinate mapped = { 0, (uint16_t)x*stride, (uint16_t)y*stride, 0 };
  float mval = -FLT_MAX;
  for ( int j = 0; j < kernel_size; ++j ){
    for ( int i = 0; i < kernel_size; ++i ){

      int index =
      b * (in_size_z * in_size_x * in_size_y) +
      z * (in_size_x * in_size_y) +
      (mapped.x + i) * (in_size_x) +
      (mapped.y);

      float v = in[index];
      if ( v > mval ){
        mval = v;
      }
    }
  }
  out[id] = mval;

  /*
  for ( int b = 0; b < in.size.b; ++b ){
    for ( int z = 0; z < out.size.z; ++z ){
      for ( int y = 0; y < out.size.y; ++y ){
        for ( int x = 0; x < out.size.x; ++x ){
          TensorCoordinate mapped = { 0, (uint16_t)x*stride, (uint16_t)y*stride, 0 };
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

__global__ void calcMaxPoolBackwardGPU(float *in1, float *in2, float *in3, float* out)
{
}

void maxPoolForwardGPU(TensorObject<float> *in, TensorObject<float> *out, int stride, int kernel_size)
{
  float *d_in, *d_out;
  int in_N = in.size.b * in.size.x * in.size.y * in.size.z;
  int N = out.size.b * out.size.x * out.size.y * out.size.z;
  cudaMalloc(&d_in,  in_N*sizeof(float));
  cudaMalloc(&d_out, N*sizeof(float));
  cudaMemcpy(d_in,  in.data,  in_N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out.data, N*sizeof(float), cudaMemcpyHostToDevice);
  dim3 grid = cudaGridSize(N);
  calcLeakyReluForwardGPU<<<grid, BLOCK>>>(d_in, d_out, in.size.x, in.size.y, in.size.z, out.size.x, out.size.y, out.size.z, strinde, kernel_size);
  cudaMemcpy(out.data, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

void maxPoolBackwardGPU(float *data_in1, float *data_in2, float *data_in3, float *data_out, int N)
{
  float *d_in1, *d_in2, *d_in3, *d_out;
  cudaMalloc(&d_in1, N*sizeof(float));
  cudaMalloc(&d_in2, N*sizeof(float));
  cudaMalloc(&d_in3, N*sizeof(float));
  cudaMalloc(&d_out, N*sizeof(float));

  cudaMemcpy(d_in1, data_in1, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, data_in2, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in3, data_in3, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, data_out, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid = cudaGridSize(N);

  calcLeakyReluBackwardGPU<<<grid, BLOCK>>>(d_in1, d_in2, d_in3, d_out);

  cudaMemcpy(data_in3, d_in3, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(data_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace gpu
