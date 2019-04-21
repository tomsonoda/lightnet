#include <stdio.h>
#include "CudaObject.h"

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

void leakyReluForwardGPU(float *data_in, float *data_out, int N)
{
  float *d_in, *d_out;
  cudaMalloc(&d_in,  N*sizeof(float));
  cudaMalloc(&d_out, N*sizeof(float));

  cudaMemcpy(d_in,  data_in,  N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, data_out, N*sizeof(float), cudaMemcpyHostToDevice);

  CudaObject *cuda = CudaObject();
  dim3 grid = cuda->cudaGridSize(N);

  calcLeakyReluForwardGPU<<<grid, BLOCK>>>(d_in, d_out);

  cudaMemcpy(data_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

// gpu_cuda::leakyReluBackwardGPU(in.data, dz_next_layer.data, dz_in.data, dz.data, data_size);
void leakyReluBackwardGPU(float *data_in1, float *data_in2, float *data_in3, float *data_out, int N)
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

  CudaObject *cuda = CudaObject();
  dim3 grid = cuda->cudaGridSize(N);

  calcLeakyReluBackwardGPU<<<grid, BLOCK>>>(d_in1, d_in2, d_in3, d_out);

  cudaMemcpy(data_in3, d_in3, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(data_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace gpu
