#include <stdio.h>
#pragma once
__device__ inline void cudaAddFirstArrayToSecondArray( float * dz_next_layer, float *dz_in, int N)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if (id<N){
    dz_in[id] += dz_next_layer[id];
  }
}

struct range_t
{
  int min_x, min_y;
  int max_x, max_y;
};

__device__ inline int normalize_range( float f, int max, bool lim_min )
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

__device__ inline range_t map_to_output( int x, int y, int dz_in_size_x, int dz_in_size_y, int kernel_size, int stride )
{
  float a = x;
  float b = y;
  float stride_inv = 1.0/stride;
  return
  {
    normalize_range( (a - kernel_size + 1) * stride_inv, dz_in_size_x, true ),
    normalize_range( (b - kernel_size + 1) * stride_inv, dz_in_size_y, true ),
    normalize_range( a * stride_inv, dz_in_size_x, false ),
    normalize_range( b * stride_inv, dz_in_size_y, false )
  };
}
