#include <stdio.h>
#define BLOCK 512

namespace gpu_cuda {

__global__ void calcLeakyReluForwardGPU(float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  float v = x[i];
  if ( v < 0 ){
    v = 0.1 * v;
  }
  y[i] = v;
}

__global__ void calcLeakyReluBackwardGPU(float *in1, float *in2, float *in3, float* out)
{
  // in1 in.data
  // in2 dz_next_layer.data
  // in3 dz_in.data
  // out dz
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  in3[i] += in2[i];
  out[i] +=  (in1[i] < 0) ? (0.1) : in3[i];
}

dim3 cudaGridSize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

void leakyReluForwardGPU(float *data_in, float *data_out, int N)
{
  float *d_in, *d_out;
  cudaMalloc(&d_in,  N*sizeof(float));
  cudaMalloc(&d_out, N*sizeof(float));

  cudaMemcpy(d_in,  data_in,  N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, data_out, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 block (BLOCK, 1, 1);
  dim3 grid  cudaGridSize(N);
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

  dim3 block (BLOCK, 1, 1);
  dim3 grid  (N / block.x, 1, 1);
  calcLeakyReluBackwardGPU<<<grid, BLOCK>>>(d_in1, d_in2, d_in3, d_out);

  cudaMemcpy(data_in3, d_in3, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(data_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace gpu
