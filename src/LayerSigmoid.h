#pragma once
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	float *cudaMakeArray( float *cpu_array, int N );
}
#endif

#pragma pack(push, 1)
struct LayerSigmoid
{
	LayerType type = LayerType::sigmoid;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;

	unsigned in_total_size;
	LayerSigmoid( TensorSize in_size )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		in_total_size = in_size.b *in_size.x *in_size.y *in_size.z;
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
		/*
		for ( int i = 0; i < in_total_size; ++i ){
			out.data[i] = activator_function(in.data[i]);
		}
		*/
	}

	void updateWeightsGPU()
	{
	}

	void backwardGPU( float* dz_next_layer, float *dz )
	{
		this->gpu_dz = dz;
		backwardGPU( dz_next_layer );
	}

	void backwardGPU( float* dz_next_layer )
	{
		/*
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		for ( int i = 0; i < in_total_size; ++i ){
			dz.data[i] += activator_derivative( in.data[i] ) * dz_in.data[i];
		}
		*/
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

#else

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for ( int i = 0; i < in_total_size; ++i ){
			out.data[i] = activator_function(in.data[i]);
		}
	}

	void updateWeights()
	{
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		for ( int i = 0; i < in_total_size; ++i ){
			dz.data[i] += activator_derivative( in.data[i] ) * dz_in.data[i];
		}
	}

#endif

	float activator_function( float x )
	{
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig;
	}

	float activator_derivative( float x )
	{
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig * (1 - sig);
	}

};
#pragma pack(pop)
