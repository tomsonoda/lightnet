#pragma once
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	void cudaMakeArray(float *gpu_o, int N);
	void leakyReluForwardGPU(float *in, float *out, int N);
	void leakyReluBackwardGPU(  float *dz_next_layer, float *gpu_dz_in, float *gpu_dz, float *gpu_in, int data_size );
} //namespace gpu
#endif

#pragma pack(push, 1)
struct LayerLeakyReLU
{
	LayerType type = LayerType::leaky_relu;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;
	float *gpu_dz_next_layer;

	int data_size;

	LayerLeakyReLU( TensorSize in_size
	)
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		data_size = in_size.b * in_size.x * in_size.y * in_size.z;

#ifdef GPU_CUDA
		gpu_cuda::cudaMakeArray(gpu_dz, data_size);
		gpu_cuda::cudaMakeArray(gpu_in, data_size);
		gpu_cuda::cudaMakeArray(gpu_out, data_size);
		gpu_cuda::cudaMakeArray(gpu_dz_in, data_size);
		gpu_cuda::cudaMakeArray(gpu_dz_next_layer, data_size);
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
		gpu_cuda::leakyReluForwardGPU( gpu_in, gpu_out, data_size );
	}

	void updateWeightsGPU()
	{
	}

	void backwardGPU( float* dz_next_layer )
	{
		this->gpu_dz_next_layer = dz_next_layer;
		gpu_cuda::leakyReluBackwardGPU( this->gpu_dz_next_layer, gpu_dz_in, gpu_dz, gpu_in, data_size );
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
				v = 0.01 * v;
			}
			out.data[i] = v;
		}
	}

	void updateWeights()
	{
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		// for( int i = 0; i < data_size ; ++i ){
		// 	dz_in.data[i] += dz_next_layer.data[i];
		// }
		// for( int i = 0; i < data_size; ++i ){
		// 	dz.data[i] +=  (in.data[i] < 0) ? (0.01) : (1.0 * dz_in.data[i]);
		// }
		for( int i = 0; i < data_size ; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
			dz.data[i] +=  (in.data[i] < 0) ? (0.01) : dz_in.data[i];
		}
	}

#endif

};
#pragma pack(pop)
