#include <stdio.h>
#include <float.h>
#include "CudaObject.h"

namespace gpu_cuda {

__global__ void calcSoftmaxMaxForwardGPU(float *array, float *max, int *mutex, unsigned n)
{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int offset = 0;

  __shared__ float cache[256];

  float temp = -1.0;
  while(index + offset < n){
    temp = fmaxf(temp, array[index + offset]);
    offset += stride;
  }

  cache[threadIdx.x] = temp;
  __syncthreads();

  unsigned int prev_i = blockDim.x;
  unsigned int i = blockDim.x / 2;
  while ( i!=0 ){
    if(threadIdx.x < i){
        cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
    }
    if(prev_i%2 != 0){
      cache[0] = fmaxf(cache[0], cache[prev_i-1]);
    }
    __syncthreads();
    i /= 2;
  }

  if( threadIdx.x == 0 ){
    while( atomicCAS(mutex, 0, 1) != 0 );
    *max = fmaxf(*max, cache[0]);
    atomicExch(mutex, 0);
  }
  /* original
  for ( int b = 0; b < in.size.b; ++b ){
    float max_v = 0.0;
    for ( int i = 0; i < in.size.x; ++i ){
      float v = in( b, i, 0, 0 );
      if(v>max_v){
        max_v = v;
      }
    }
  }
  */
}

__global__ void calcSoftmaxSumForwardGPU(float *array, float *out, float *max, float *sum, int *mutex, unsigned n)
{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int offset = 0;

  __shared__ float cache[512];

  float temp = 0.0;
  while(index + offset < n){
    float v = exp(array[index + offset] - *max);
    out[index + offset] = v;
    temp = temp + v;
    offset += stride;
  }

  cache[threadIdx.x] = temp;
  __syncthreads();

  unsigned int prev_i = blockDim.x;
  unsigned int i = blockDim.x / 2;

  while ( i!=0 ){
    if(threadIdx.x < i){
        cache[threadIdx.x] = cache[threadIdx.x] + cache[threadIdx.x + i];
    }
    if(prev_i%2 != 0){
      cache[0] = cache[0] + cache[prev_i-1];
    }
    __syncthreads();
    prev_i = i;
    i /= 2;
  }

  if( threadIdx.x == 0 ){
    while( atomicCAS(mutex, 0, 1) != 0 );
    *sum = *sum + cache[0];
    atomicExch(mutex, 0);
  }

  /* original
  float sum = 0.0;
  for ( int i = 0; i < in.size.x; ++i ){
    float v = in( b, i, 0, 0 );
    v = exp(v - max_v);
    out( b, i, 0, 0 ) = v;
    sum += v;
  }
  */
}

__global__ void calcSoftmaxDivForwardGPU(float *out, float *sum, unsigned int n)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if(id<n && *sum>0.0){
    out[id] = out[id] / *sum;
  }

  /* original
  for ( int i = 0; i < in.size.x; ++i ){
    out( b, i, 0, 0 ) = out( b, i, 0, 0 ) / sum;
  }
  */
}

__global__ void calcSoftmaxBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, unsigned int n )
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if ( id < n ){
    dz_in[id] += dz_next_layer[id];
    dz[id] +=  dz_in[id];
  }

  /* original
  for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
    dz_in.data[i] += dz_next_layer.data[i];
  }

  for ( int i = 0; i < in.size.b * in.size.x * in.size.y * in.size.z; ++i ){
    dz.data[i] += dz_in.data[i];
  }
  */
}

void softmaxForwardGPU( float *in, float *out, int batch_size, int in_size_x )
{
  CudaObject cuda = CudaObject();
  int elements = batch_size * in_size_x;
  dim3 grid = cuda.cudaGridSize( elements );
  int *d_mutex;

  float *d_max;
  float *d_sum;
  cudaMalloc( (void **)&d_max, sizeof(float));
  cudaMalloc( (void **)&d_sum, sizeof(float));
  cudaMalloc((void**)&d_mutex, sizeof(int));

  cudaMemset(d_max, 0, sizeof(float));
  cudaMemset(d_sum, 0, sizeof(float));

  cudaMemset(d_mutex, 0, sizeof(int));              // 0 means unlocked.
  calcSoftmaxMaxForwardGPU<<<1, elements, elements * sizeof(float) >>>( in, d_max, d_mutex, elements );
  cudaMemset(d_mutex, 0, sizeof(int));              // 0 means unlocked.
  calcSoftmaxSumForwardGPU<<<1, elements, elements * sizeof(float) >>>( in, out, d_max, d_sum, d_mutex, elements );
  calcSoftmaxDivForwardGPU<<<grid, BLOCK>>>( out, d_sum, elements );

  cudaFree(d_max);
  cudaFree(d_sum);
  cudaFree(d_mutex);
}

void softmaxBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, int N )
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcSoftmaxBackwardGPU<<<grid, BLOCK>>>( dz_next_layer, dz_in, dz, N );
}

} // namespace gpu
