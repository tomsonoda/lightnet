#include <stdio.h>

#ifdef GPU

namespace gpu_cuda {

  void cudaMakeArray(float* gpu_o, int N)
  {
    cudaMalloc(&gpu_o, N*sizeof(float));
  }

} // namespace gpu

#endif
