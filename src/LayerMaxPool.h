#pragma once
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	float *cudaMakeArray( float *cpu_array, int N );
	void maxPoolForwardGPU(float *in, float *out, int in_size_x, int in_size_y, int in_size_z, int out_size_b, int out_size_x, int out_size_y, int out_size_z, int kernel_size, int stride );
	void maxPoolBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, float *out, int batch_size, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int kernel_size, int stride );
} //namespace gpu
#endif


#pragma pack(push, 1)
struct LayerMaxPool
{
	LayerType type = LayerType::max_pool;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;

	uint16_t stride;
	uint16_t kernel_size;

	LayerMaxPool( uint16_t stride, uint16_t kernel_size, TensorSize in_size )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out(
			in_size.b,
			(in_size.x - kernel_size) / stride + 1,
			(in_size.y - kernel_size) / stride + 1,
			in_size.z
		),
		dz_in(
			in_size.b,
			(in_size.x - kernel_size) / stride + 1,
			(in_size.y - kernel_size) / stride + 1,
			in_size.z
		)

	{
		this->stride = stride;
		this->kernel_size = kernel_size;

#ifdef GPU_CUDA
		int data_size = in_size.b * in_size.x * in_size.y * in_size.z;
		gpu_dz = gpu_cuda::cudaMakeArray( dz, data_size );
		gpu_in = gpu_cuda::cudaMakeArray( in, data_size );
		int dz_in_size = dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z;
		gpu_out = gpu_cuda::cudaMakeArray( out, dz_in_size );
		gpu_dz_in = gpu_cuda::cudaMakeArray( dz_in, dz_in_size );
#endif

		assert( (float( in_size.x - kernel_size ) / stride + 1)
				==
				((in_size.x - kernel_size) / stride + 1) );

		assert( (float( in_size.y - kernel_size ) / stride + 1)
				==
				((in_size.y - kernel_size) / stride + 1) );

	}

	TensorCoordinate map_to_input( TensorCoordinate out, int z )
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	struct range_t
	{
		int min_x, min_y;
		int max_x, max_y;
	};

	int normalize_range( float f, int max, bool lim_min )
	{
		if ( f <= 0 ){
			return 0;
		}
		max -= 1;
		if ( f >= max ){
			return max;
		}

		if ( lim_min ){ // left side of inequality
			return ceil( f );
		}else{
			return floor( f );
		}
	}

	range_t map_to_output( int x, int y )
	{
		float a = x;
		float b = y;
		float stride_inv = 1.0/stride;
		return
		{
			normalize_range( (a - kernel_size + 1) * stride_inv, out.size.x, true ),
			normalize_range( (b - kernel_size + 1) * stride_inv, out.size.y, true ),
			normalize_range( a * stride_inv, out.size.x, false ),
			normalize_range( b * stride_inv, out.size.y, false )
		};
	}

#ifdef GPU_CUDA

	void forwardGPU( float* in )
	{
		this->gpu_in = in;
		forwardGPU();
	}

	void forwardGPU()
	{
		gpu_cuda::maxPoolForwardGPU(gpu_in, gpu_out, in.size.x, in.size.y, in.size.z, out.size.b, out.size.x, out.size.y, out.size.z, kernel_size, stride);
	}

	void updateWeightsGPU()
	{
	}

	void backwardGPU( float *dz_next_layer )
	{
		gpu_cuda::maxPoolBackwardGPU( dz_next_layer, gpu_dz_in, gpu_dz, gpu_in, gpu_out, dz.size.b, dz.size.x, dz.size.y, dz.size.z, dz_in.size.x, dz_in.size.y, dz_in.size.z, kernel_size, stride );
	}

#else
// CPU

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for ( int b = 0; b < in.size.b; ++b ){
			for ( int z = 0; z < out.size.z; ++z ){
				for ( int y = 0; y < out.size.y; ++y ){
					for ( int x = 0; x < out.size.x; ++x ){
						TensorCoordinate mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
						float mval = -FLT_MAX;
						for ( int j = 0; j < kernel_size; ++j ){
							for ( int i = 0; i < kernel_size; ++i ){
								float v = in( b, mapped.x + i, mapped.y + j, z );
								if ( v > mval ){
									mval = v;
								}
							}
						}
						out( b, x, y, z ) = mval;
					}
				}
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
		for ( int b = 0; b < in.size.b; ++b ){
			for ( int y = 0; y < in.size.y; ++y ){
				for ( int x = 0; x < in.size.x; ++x ){
					range_t rn = map_to_output( x, y );
					for ( int z = 0; z < in.size.z; ++z ){
						float sum_error = 0;
						float in_value = in( b, x, y, z );
						for ( int j = rn.min_y; j <= rn.max_y; ++j ){
							for ( int i = rn.min_x; i <= rn.max_x; ++i ){
								int is_max = in_value == out( b, i, j, z ) ? 1 : 0;
								sum_error += is_max * dz_in( b, i, j, z );
							}
						}
						dz( b, x, y, z ) += sum_error;
					}
				}
			}
		}
	}

#endif

};
#pragma pack(pop)
