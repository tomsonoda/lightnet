#ifndef CUDA_H
#define CUDA_H

#define BLOCK 512

#include <stdio.h>

#ifdef GPU

#pragma pack(push, 1)
struct CudaObject
{
  void cudaMakeArray(float* gpu_out, int N){
    cudaMalloc(&gpu_out, N*sizeof(float));
  }

  dim3 cudaGridSize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d  (x, y, 1);
    return d;
  }
};
#pragma pack(pop)

#endif
#endif
