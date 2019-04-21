#include <stdio.h>

namespace gpu_cuda {

__global__ void calcMaxPoolForwardGPU(float *in,float *out,
  int in_size_x, int in_size_y, int in_size_z,
  int size_x, int size_y, int size_z,
  int stride, int kernel_size)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int id_out = id;

  int x = id % size_x;
  id /= size_x;
  int y = id % size_y;
  id /= size_y;
  int z = id % size_z;
  id /= size_z;
  int b = id;

  int mapped_x = x * stride;
  int mapped_y = y * stride;
  float mval = -100000.0;
  for ( int j = 0; j < kernel_size; ++j ){
    for ( int i = 0; i < kernel_size; ++i ){

      int index =
      b * (in_size_z * in_size_x * in_size_y) +
      z * (in_size_x * in_size_y) +
      (mapped_x + i) * (in_size_x) +
      (mapped_y);

      float v = in[index];
      if ( v > mval ){
        mval = v;
      }
    }
  }
  out[id_out] = mval;
}

__global__ void calcMaxPoolBackwardGPU(float *in1, float *in2, float *in3, float* out)
{
}

void maxPoolForwardGPU(float *data_in, float *data_out,
  int in_size_b, int in_size_x, int in_size_y, int in_size_z,
  int out_size_b, int out_size_x, int out_size_y, int out_size_z,
  int stride, int kernel_size)
{
  float *d_in, *d_out;
  int in_N = in_size_b * in_size_x * in_size_y * in_size_z;
  int N = out_size_b * out_size_x * out_size_y * out_size_z;
  cudaMalloc(&d_in,  in_N*sizeof(float));
  cudaMalloc(&d_out, N*sizeof(float));
  cudaMemcpy(d_in,  data_in,  in_N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, data_out, N*sizeof(float), cudaMemcpyHostToDevice);
  dim3 grid = cudaGridSize(N);
  calcMaxPoolForwardGPU<<<grid, BLOCK>>>(d_in, d_out, in_size_x, in_size_y, in_size_z, out_size_x, out_size_y, out_size_z, stride, kernel_size);
  cudaMemcpy(data_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

void maxPoolBackwardGPU(float *data_in1, float *data_in2, float *data_in3, float *data_out, int N)
{
}

} // namespace gpu
