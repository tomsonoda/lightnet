#define BLOCK 512
#pragma once
#include <stdio.h>

#ifdef GPU

namespace gpu_cuda {
  dim3 cudaGridSize(size_t n)
  {
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

  void cudaMakeArray(float* gpu_out, int N)
  {
    cudaMalloc(&gpu_out, N*sizeof(float));
  }
}

#endif
