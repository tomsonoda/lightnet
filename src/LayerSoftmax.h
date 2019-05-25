#pragma once
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	void cudaGetArray( float *cpu_array, float *gpu_array, int N );
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
				gpu_dz    = gpu_cuda::cudaMakeArray( NULL, d_size );
				gpu_in    = gpu_cuda::cudaMakeArray( NULL, d_size );
				gpu_out   = gpu_cuda::cudaMakeArray( NULL, d_size );
				gpu_dz_in = gpu_cuda::cudaMakeArray( NULL, d_size );
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
		printf("softmax forward gpu\n");
		// gpu_cuda::cudaGetArray( out.data, gpu_out, in.size.b * in.size.x * in.size.y * in.size.z );
		gpu_cuda::softmaxForwardGPU( gpu_in, gpu_out, in.size.b, in.size.x );
		printf("softmax forward gpu finish\n");
	}

	void updateWeightsGPU()
	{
	}

	void backwardGPU( float* dz_next_layer )
	{
		int in_size = in.size.b * in.size.x * in.size.y * in.size.z;
		gpu_cuda::softmaxBackwardGPU( dz_next_layer, gpu_dz_in, gpu_dz, in_size );
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
