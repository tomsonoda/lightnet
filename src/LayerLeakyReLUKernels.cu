#include <stdio.h>
#define BLOCK 512

namespace gpu_cuda {

// dim3 cuda_gridsize(size_t n){
//   size_t k = (n-1) / BLOCK + 1;
//   size_t x = k;
//   size_t y = 1;
//   if(x > 65535){
//       x = ceil(sqrt(k));
//       y = (n-1)/(x*BLOCK) + 1;
//   }
//   //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
//   return {x, y, 1};
// }

__global__ void calc(float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  float v = x[i];
  if ( v < 0 ){
    v = 0.1 * v;
  }
  y[i] = v;
}

void leakyReluForwardGPU(float *data_in, float *data_out, int N)
{
  float *d_in, *d_out;
  cudaMalloc(&d_in,  N*sizeof(float));
  cudaMalloc(&d_out, N*sizeof(float));

  cudaMemcpy(d_in,  data_in,  N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, data_out, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 block (BLOCK, 1, 1);
  dim3 grid  (N / block.x, 1, 1);
  calc<<<grid, BLOCK>>>(d_in, d_out);

  cudaMemcpy(data_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace gpu
