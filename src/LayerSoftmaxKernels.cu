#include <stdio.h>
#include <float.h>
#include "CudaObject.h"

namespace gpu_cuda {

__device__ float atomicMaxf(float* address, float val)
{
  int *address_as_int =(int*)address;
  int old = *address_as_int, assumed;
  while (val > __int_as_float(old)) {
      assumed = old;
      old = atomicCAS(address_as_int, assumed,
                      __float_as_int(val));
      }
  return __int_as_float(old);
}

__global__ void calcSoftmaxMaxForwardGPU(float *in, float *d_max, int elements)
{
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid;
  shared[tid] = -FLT_MAX;  // 1

  if (gid < elements)
    shared[tid] = in[gid];
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if (tid < s && gid < elements)
          shared[tid] = max(shared[tid], shared[tid + s]);  // 2
          __syncthreads();
      }

 // what to do now?
 // option 1: save block result and launch another kernel
 if (tid == 0)
    d_max[blockIdx.x] = shared[tid]; // 3
 // option 2: use atomics
 if (tid == 0)
   atomicMaxf(d_max, shared[0]);

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

__global__ void calcSoftmaxSumForwardGPU(float *in, float *out, float *d_max, int elements)
{
  // int blockID  = blockIdx.x;
  // int nBlocks  = gridDim.x;
  // int threadID = threadIdx.x;
  // int nThrads  = blockDim.x;



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

  /* original
  for ( int i = 0; i < in.size.x; ++i ){
    out( b, i, 0, 0 ) = out( b, i, 0, 0 ) / sum;
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
  calcSoftmaxMaxForwardGPU<< <grid, BLOCK, batch_size * in_size_x * sizeof(float) >>>( in, odata, batch_size * in_size_x );
  calcSoftmaxSumForwardGPU<<<grid, BLOCK, batch_size * in_size_x * sizeof(float) >>>( in, out, odata, batch_size * in_size_x );
  calcSoftmaxDivForwardGPU<<<grid, BLOCK>>>( out, odata, batch_size, in_size_x );
  cudaFree(odata);
}

void softmaxBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, int N )
{
  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcSoftmaxBackwardGPU<<<grid, BLOCK>>>( dz_next_layer, dz_in, dz );
}

} // namespace gpu
