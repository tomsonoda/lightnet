#pragma once
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	float *cudaMakeArray( float *cpu_array, int N );
	void leakyReluForwardGPU(float *in, float *out, int N);
	void leakyReluBackwardGPU( float *gpu_dz_next_layer, float *gpu_dz_in, float *gpu_dz, float *gpu_in, int data_size );
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
		gpu_dz = gpu_cuda::cudaMakeArray( dz.data, data_size );
		gpu_in = gpu_cuda::cudaMakeArray( in.data, data_size );
		gpu_out = gpu_cuda::cudaMakeArray( out.data, data_size );
		gpu_dz_in = gpu_cuda::cudaMakeArray( dz_in.data, data_size );
#endif

	}

#ifdef GPU_CUDA

	void forwardGPU( float *in, float *out )
	{
		gpu_in = in;
		gpu_out = out;
		forwardGPU();
	}

	void forwardGPU()
	{
		gpu_cuda::leakyReluForwardGPU( gpu_in, gpu_out, data_size );
	}

	void updateWeightsGPU()
	{
	}

	void backwardGPU( float* dz_next_layer, float *dz, float *dz_in )
	{
		this->gpu_dz = dz;
		this->gpu_dz_in = dz_in;
		backwardGPU( dz_next_layer );
	}

	void backwardGPU( float* dz_next_layer )
	{
		gpu_cuda::leakyReluBackwardGPU( dz_next_layer, gpu_dz_in, gpu_dz, gpu_in, data_size );
	}

	TensorObject<float> getOutFromGPU(){
		gpu_cuda::cudaGetArray( out.data, gpu_out, out.size.b*out.size.x*out.size.y*out.size.z );
		return out;
	}

	void clearArrayGPU(float *dz_)
	{
		this->gpu_dz = dz_;
		gpu_cuda::cudaClearArray( gpu_dz_in, dz_in.size.b*dz_in.size.x*dz_in.size.y*dz_in.size.z );
		gpu_cuda::cudaClearArray( gpu_dz, dz.size.b*dz.size.x*dz.size.y*dz.size.z );
	}

#endif

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for( int i = 0; i < data_size; ++i ){
			float v = in.data[i];
			if( v < 0 ){
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

// #endif

};
#pragma pack(pop)
