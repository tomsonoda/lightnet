#include <stdio.h>
#include "CudaObject.h"

namespace gpu_cuda {

__global__ void calcConvolutionalForwardGPU(float *x, float *y)
{
}

__global__ void calcConvolutionalBackwardGPU(float *in1, float *in2, float *in3, float* out)
{
}

void convolutionalForwardGPU(float *data_in, float *data_out, int N)
{
}

void convolutionalBockwardGPU(float *data_in1, float *data_in2, float *data_in3, float *data_out, int N)
{
}

} // namespace gpu
