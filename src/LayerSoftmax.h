#pragma once
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	float *cudaMakeArray( float *cpu_array, int N );
	void softmaxForwardGPU(float *in, float *out, int batch_size, int in_size );
	void softmaxBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, int N );
}
#endif

#pragma pack(push, 1)
struct LayerSoftmax
{
	LayerType type = LayerType::softmax;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;

	LayerSoftmax( TensorSize in_size )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		#ifdef GPU_CUDA
				int d_size = in_size.b * in_size.x * in_size.y * in_size.z;
				// gpu_dz    = gpu_cuda::cudaMakeArray( NULL, d_size );
				// gpu_in    = gpu_cuda::cudaMakeArray( NULL, d_size );
				// gpu_out   = gpu_cuda::cudaMakeArray( NULL, d_size );
				gpu_dz_in = gpu_cuda::cudaMakeArray( NULL, d_size );
		#endif
 	}
#ifdef GPU_CUDA

	void forwardGPU( float *in, float *out )
	{
		gpu_in = in;
		gpu_out = out;
		// forward();
		forwardGPU();
	}

	void forwardGPU()
	{
		gpu_cuda::softmaxForwardGPU( gpu_in, gpu_out, in.size.b, in.size.x );
	}

	void updateWeightsGPU()
	{
	}

	void backwardGPU( float* dz_next_layer, float *dz, float *dz_in )
	{
		this->gpu_dz = dz;
		this->gpu_dz_in = dz_in;
		backwardGPU( dz_next_layer );
		// TensorObject<float> dz_next_layer_cpu = TensorObject<float>(out.size.b, out.size.x, out.size.y, out.size.z);
		// gpu_cuda::cudaGetArray( dz_next_layer_cpu.data, dz_next_layer, dz_next_layer_cpu.size.b*dz_next_layer_cpu.size.x*dz_next_layer_cpu.size.y*dz_next_layer_cpu.size.z );
		// backward( dz_next_layer_cpu );
	}

	void backwardGPU( float* dz_next_layer )
	{
		int in_size = in.size.b * in.size.x * in.size.y * in.size.z;
		gpu_cuda::softmaxBackwardGPU( dz_next_layer, gpu_dz_in, gpu_dz, in_size );
	}

	TensorObject<float> getOutFromGPU(){
		gpu_cuda::cudaGetArray( out.data, gpu_out, out.size.b*out.size.x*out.size.y*out.size.z );
		return out;
	}

	TensorObject<float> getDzFromGPU(){
		gpu_cuda::cudaGetArray( dz.data, gpu_dz, dz.size.b*dz.size.x*dz.size.y*dz.size.z );
		return dz;
	}

	void clearArrayGPU(float *dz_)
	{
		this->gpu_dz = dz_;
		gpu_cuda::cudaClearArray( gpu_dz_in, dz_in.size.b*dz_in.size.x*dz_in.size.y*dz_in.size.z );
		gpu_cuda::cudaClearArray( gpu_dz, dz.size.b*dz.size.x*dz.size.y*dz.size.z );
		dz_in.clear();
		dz.clear();
	}

#else

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for ( int b = 0; b < in.size.b; ++b ){

			float max_v = 0.0;
			for ( int i = 0; i < in.size.x; ++i ){
				float v = in( b, i, 0, 0 );
				if(v>max_v){
					max_v = v;
				}
			}

			float sum = 0.0;
			for ( int i = 0; i < in.size.x; ++i ){
				float v = in( b, i, 0, 0 );
				v = exp(v - max_v);
				out( b, i, 0, 0 ) = v;
				sum += v;
			}

			for ( int i = 0; i < in.size.x; ++i ){
				out( b, i, 0, 0 ) = out( b, i, 0, 0 ) / sum;
			}

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

		for ( int i = 0; i < in.size.b * in.size.x * in.size.y * in.size.z; ++i ){
			dz.data[i] += dz_in.data[i];
		}
	}
#endif
};
#pragma pack(pop)
