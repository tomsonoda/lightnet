#pragma once
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	float *cudaMakeArray( float *cpu_array, int N );
	void reluForwardGPU( float *gpu_in, float *gpu_out, int N);
	void reluBackwardGPU( float *dz_next_layer, float *gpu_dz_in, float *gpu_dz, float *in, int N );
}
#endif

#pragma pack(push, 1)
struct LayerReLU
{
	LayerType type = LayerType::relu;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;

	unsigned data_size;

	LayerReLU( TensorSize in_size )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		data_size = in_size.b * in_size.x * in_size.y * in_size.z;

#ifdef GPU_CUDA
		gpu_dz = gpu_cuda::cudaMakeArray( dz.data, data_size );
		gpu_in = gpu_cuda::cudaMakeArray( in.data, data_size );
		gpu_out = gpu_cuda::cudaMakeArray( out.data, data_size );
		gpu_dz_in = gpu_cuda::cudaMakeArray( dz_in.data, data_size );
#endif

	}

#ifdef GPU_CUDA

	void forwardGPU( float* in )
	{
		this->gpu_in = in;
		forwardGPU();
	}

	void forwardGPU()
	{
		gpu_cuda::reluForwardGPU( gpu_in, gpu_out, data_size );
	}

	void updateWeightsGPU()
	{
	}

	void backwardGPU( float* dz_next_layer )
	{
		gpu_cuda::reluBackwardGPU( dz_next_layer, gpu_dz_in, gpu_dz, gpu_in, data_size );
	}

#else

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for( int i = 0; i < data_size; ++i ){
			float v = in.data[i];
			if ( v < 0 ){
				v = 0;
			}
			out.data[i] = v;
		}
	}

	void updateWeights()
	{
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		for( int i = 0; i < data_size; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
			dz.data[i] +=  (in.data[i] < 0) ? (0) : (1.0 * dz_in.data[i]);
		}
	}

#endif

};

#pragma pack(pop)
