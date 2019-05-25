#include <stdio.h>
#include "CudaObject.h"

namespace gpu_cuda {

__global__ void calcSoftmaxMaxForwardGPU(float *in, float *odata, int batch_size, int in_size_x)
{
  extern __shared__ int sdata[];
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;
  sdata[tid] = (id < batch_size * in_size_x) ? in[id] : 0;

  if(id<batch_size * in_size_x){
    __syncthreads();
    for (unsigned int s=1; s<blockDim.x && s<(batch_size * in_size_x); s*=2) {
      if (tid % (2*s) == 0) {
        sdata[tid] =  sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s];
      }
      __syncthreads();
    }
  }

  if (tid == 0){
    odata[0] = sdata[0];
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

__global__ void calcSoftmaxSumForwardGPU(float *in, float *out, float *odata, int batch_size, int in_size_x)
{
  // int blockID  = blockIdx.x;
  // int nBlocks  = gridDim.x;
  // int threadID = threadIdx.x;
  // int nThrads  = blockDim.x;

  extern __shared__ int sdata[];
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;
  sdata[tid] = (id < batch_size * in_size_x) ? in[id] : 0;

  if(id<batch_size * in_size_x){
    __syncthreads();
    for (unsigned int s=1; s<blockDim.x && s<(batch_size * in_size_x); s*=2) {
      if (tid % (2*s) == 0) {
        sdata[tid] += expf( sdata[tid + s] - odata[0] );
      }
      __syncthreads();
    }
  }

  out[id] = expf( in[id] - odata[0] );

  // sum.
  if (tid == 0){
    odata[0] = sdata[0];
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

__global__ void calcSoftmaxDivForwardGPU(float *out, float *odata, int batch_size, int in_size_x)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if(id<batch_size * in_size_x && odata[0]>0.0){
    out[id] = out[id] / odata[0];
  }
}


__global__ void calcSoftmaxForwardGPU(float *in, float *out, int batch_size, int in_size_x)
{
  // int blockID  = blockIdx.x;
  // int nBlocks  = gridDim.x;
  // int threadID = threadIdx.x;
  // int nThrads  = blockDim.x;

  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if(id<batch_size){
    float max_v = 0.0;

    for ( int i = 0; i < in_size_x; ++i ){
      float v = in[id + i];
      if(v>max_v){
        max_v = v;
      }
    }

    float sum = 0.0;
    for ( int i = 0; i < in_size_x; ++i ){
      float v = in[id + i];
      v = expf(v - max_v);
      out[id + i] = v;
      sum += v;
    }

    if (sum > 0.0){
      for ( int i = 0; i < in_size_x; ++i ){
        float v = out[id + i] / sum;
        out[id + i] = v;
      }
    }
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

    float sum = 0.0;
    for ( int i = 0; i < in.size.x; ++i ){
      float v = in( b, i, 0, 0 );
      v = exp(v - max_v);
      out( b, i, 0, 0 ) = v;
      sum += v;
    }

    for ( int i = 0; i < in.size.x; ++i ){
      out( b, i, 0, 0 ) = out( b, i, 0, 0 ) / sum;
    }

  }
  */
}

__global__ void calcSoftmaxBackwardGPU( float *dz_next_layer, float *dz_in, float *dz)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  dz_in[id] += dz_next_layer[id];
  dz[id] +=  dz_in[id];

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
  dim3 grid = cuda.cudaGridSize( batch_size * in_size_x );

  float *odata;
  cudaMalloc( (void **)&odata, sizeof(float));
  calcSoftmaxMaxForwardGPU<<<grid, BLOCK>>>( in, odata, batch_size, in_size_x );
  calcSoftmaxSumForwardGPU<<<grid, BLOCK>>>( in, out, odata, batch_size, in_size_x );
  calcSoftmaxDivForwardGPU<<<grid, BLOCK>>>( out, odata, batch_size, in_size_x );

  // calcSoftmaxForwardGPU<<<grid, BLOCK>>>( in, out, batch_size, in_size_x );
}

void softmaxBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, int N )
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcSoftmaxBackwardGPU<<<grid, BLOCK>>>( dz_next_layer, dz_in, dz );
}

} // namespace gpu
