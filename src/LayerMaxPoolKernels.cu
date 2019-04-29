#include <stdio.h>
#include "CudaObject.h"

namespace gpu_cuda {

__global__ void calcMaxPoolForwardGPU(float *in,float *out,
  int in_size_x, int in_size_y, int in_size_z,
  int size_x, int size_y, int size_z,
  int stride, int kernel_size
)
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

__device__ struct range_t
{
  int min_x, min_y;
  int max_x, max_y;
};

__device__ int normalize_range( float f, int max, bool lim_min )
{
  if ( f <= 0 ){
    return 0;
  }
  max -= 1;
  if ( f >= max ){
    return max;
  }

  if ( lim_min ){ // left side of inequality
    return ceil( f );
  }else{
    return floor( f );
  }
}

__device__ range_t map_to_output( int x, int y )
{
  float a = x;
  float b = y;
  float stride_inv = 1.0/stride;
  return
  {
    normalize_range( (a - kernel_size + 1) * stride_inv, out.size.x, true ),
    normalize_range( (b - kernel_size + 1) * stride_inv, out.size.y, true ),
    normalize_range( a * stride_inv, out.size.x, false ),
    normalize_range( b * stride_inv, out.size.y, false )
  };
}

__global__ void calcMaxPoolBackwardGPU(
}

void maxPoolForwardGPU(float *data_in, float *data_out,
  float *d_in, float *d_out,
  int in_size_b, int in_size_x, int in_size_y, int in_size_z,
  int out_size_b, int out_size_x, int out_size_y, int out_size_z,
  int stride, int kernel_size)
{
  int in_N = in_size_b * in_size_x * in_size_y * in_size_z;
  int N = out_size_b * out_size_x * out_size_y * out_size_z;
  cudaMemcpy(d_in,  data_in,  in_N*sizeof(float), cudaMemcpyHostToDevice);

  CudaObject cuda = CudaObject();
  dim3 grid = cuda.cudaGridSize(N);
  calcMaxPoolForwardGPU<<<grid, BLOCK>>>(d_in, d_out, in_size_x, in_size_y, in_size_z, out_size_x, out_size_y, out_size_z, stride, kernel_size);

  cudaMemcpy(data_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
}

void maxPoolBackwardGPU(
  float *gpu_in, float *gpu_out, float *gpu_dz, float *gpu_dz_in,
  float *in, float *out, float *dz, float *dz_in,
  int in_size_b, int in_size_x, int in_size_y, int in_size_z,
  int out_size_b, int out_size_x, int out_size_y, int out_size_z,
  int stride, int kernel_size
)
{
}
} // namespace gpu
