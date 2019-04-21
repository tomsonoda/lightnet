#pragma once
#include <stdio.h>
#define BLOCK 512

namespace gpu_cuda {

__global__ void calcConvolutionalForwardGPU(float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  float v = x[i];
  if ( v < 0 ){
    v = 0.1 * v;
  }
  y[i] = v;
}

__global__ void calcConvolutionalBackwardGPU(float *in1, float *in2, float *in3, float* out)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  in3[i] += in2[i];
  out[i] +=  (in1[i] < 0) ? (0.1) : in3[i];
}

void convolutionalForwardGPU(float *data_in, float *data_out, int N)
{
  float *d_in, *d_out;
  cudaMalloc(&d_in,  N*sizeof(float));
  cudaMalloc(&d_out, N*sizeof(float));

  cudaMemcpy(d_in,  data_in,  N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, data_out, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 block (BLOCK, 1, 1);
  dim3 grid  (N / block.x, 1, 1);
  calcLeakyReluForwardGPU<<<grid, BLOCK>>>(d_in, d_out);

  cudaMemcpy(data_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

// gpu_cuda::leakyReluBackwardGPU(in.data, dz_next_layer.data, dz_in.data, dz.data, data_size);
void convolutionalBockwardGPU(float *data_in1, float *data_in2, float *data_in3, float *data_out, int N)
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

  dim3 block (BLOCK, 1, 1);
  dim3 grid  (N / block.x, 1, 1);
  calcLeakyReluBackwardGPU<<<grid, BLOCK>>>(d_in1, d_in2, d_in3, d_out);

  cudaMemcpy(data_in3, d_in3, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(data_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace gpu
