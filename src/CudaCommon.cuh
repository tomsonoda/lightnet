#include <stdio.h>
#pragma once
__global__ inline void cudaAddFirstArrayToSecondArray(float * dz_next_layer, float *dz_in)
{
  int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  dz_in[id] += dz_next_layer[id];
}
