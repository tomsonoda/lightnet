#include <stdio.h>
#include "CudaObject.h"

#ifdef GPU

namespace gpu_cuda {
  __global__ void calc(int N){
  }

  void cudaMakeArray(float* gpu_o, int N)
  {
    cudaMalloc(&gpu_o, N*sizeof(float));
  }

} // namespace gpu

#endif
