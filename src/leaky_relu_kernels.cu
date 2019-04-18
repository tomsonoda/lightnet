#include <stdio.h>

namespace gpu_cuda {

__global__

void calc(int n, float *x, float *y)
{
  for( int i = 0; i < n; ++i ){
    float v = x[i];
    if ( v < 0 ){
      v = 0.1 * v;
    }
    y[i] = v;
  }
}

void leakyReluForwardGPU(float *data_in, float *data_out, int N)
{
  float *d_in, *d_out;
  cudaMalloc(&d_in,  N*sizeof(float));
  cudaMalloc(&d_out, N*sizeof(float));

  cudaMemcpy(d_in,  data_in,  N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, data_out, N*sizeof(float), cudaMemcpyHostToDevice);

  const int threads_per_block = 2*32;
  const int blocks_per_grid = 1*12;
  calc<<<blocks_per_grid, threads_per_block>>>(N, d_in, d_out);

  cudaMemcpy(data_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++){
    printf("data_size=%d pair: %f %f\n", N, data_in, data_out);
  }
}

} // namespace gpu
