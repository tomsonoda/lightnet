#include <stdio.h>
#include "Cuda.h"

namespace gpu_cuda {

__global__ void calcLeakyReluForwardGPU(float *x, float *y)
{
  // int i = blockIdx.x*blockDim.x + threadIdx.x;
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  float v = x[id];
  if ( v < 0 ){
    v = 0.1 * v;
  }
  y[id] = v;
}

__global__ void calcLeakyReluBackwardGPU(float *in1, float *in2, float *in3, float* out)
{
  // in1 in.data
  // in2 dz_next_layer.data
  // in3 dz_in.data
  // out dz
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  in3[id] += in2[id];
  out[id] += (in1[id] < 0) ? (0.1) : in3[id];
}

void leakyReluForwardGPU(float *data_in, float *data_out, float *gpu_in, float *gpu_out, int N)
{
  cudaMemcpy(gpu_in,  data_in,  N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_out, data_out, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid = cudaGridSize(N);
  calcLeakyReluForwardGPU<<<grid, BLOCK>>>(gpu_in, gpu_out);

  cudaMemcpy(data_out, gpu_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

// gpu_cuda::leakyReluBackwardGPU(in.data, dz_next_layer.data, dz_in.data, dz.data, data_size);
void leakyReluBackwardGPU(float *data_in1, float *data_in2, float *data_in3, float *data_out,
  float *gpu_in, float *gpu_dz_next_layer, float *gpu_dz_in, float *gpu_dz,
  int N)
{
  cudaMemcpy(gpu_in, data_in1, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_dz_next_layer, data_in2, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_dz_in, data_in3, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_dz, data_out, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid = cudaGridSize(N);

  calcLeakyReluBackwardGPU<<<grid, BLOCK>>>(gpu_in, gpu_dz_next_layer, gpu_dz_in, gpu_dz);

  cudaMemcpy(data_in3, gpu_dz_in, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(data_out, gpu_dz, N*sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace gpu
