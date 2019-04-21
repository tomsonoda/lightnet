#include <stdio.h>

#ifdef GPU

namespace gpu_cuda {
  void cudaMakeArray(float* gpu_out, int N)
  {
    cudaMalloc(&gpu_out, N*sizeof(float));
  }
}

#endif
