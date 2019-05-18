#pragma once
#include <iostream>
#include <fstream>
#include "LayerObject.h"
#include "GradientObject.h"
#include "ThreadPool.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	float *cudaMakeArray( float *cpu_array, int N );
	void cudaClearArray( float *gpu_array, int N );
	void cudaMakeRandomArray(float *gpu_array, int N, int maxval );
	void convolutionForwardGPU( float *in, float *out, float *padded_in, float *filters, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int padding, int kernel_size, int stride, int filter_size );
	void convolutionBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *padded_in, float *filters, float *filter_grads, int batch_size, int dz_size_x, int dz_size_y, int dz_size_z, int dz_in_size_x, int dz_in_size_y, int dz_in_size_z, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int padding, int kernel_size, int stride, int number_filters, int filter_size );
	void convolutionUpdateWeightsGPU(float *filters, float *filter_grads, int in_size_z, int number_filters, int kernel_size, float momentum, float decay, float learning_rate);
}
#endif

#pragma pack(push, 1)
struct LayerConvolution
{
	LayerType type = LayerType::conv;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;

	TensorObject<float> padded_in;
	std::vector<TensorObject<float>> filters;
	std::vector<TensorObject<GradientObject>> filter_grads;

	float *gpu_padded_in;
	float *gpu_filters;
	float *gpu_filter_grads;

	uint16_t stride;
	uint16_t kernel_size;
	uint16_t padding;
	uint16_t filter_size;

	uint16_t number_filters;
	float lr;
	float decay;
	float momentum;
	uint16_t dz_in_size;

	LayerConvolution( uint16_t stride, uint16_t kernel_size, uint16_t number_filters, uint16_t padding, TensorSize in_size, float learning_rate, float decay, float momentum )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out(
			in_size.b,
			(in_size.x - kernel_size + 2*padding) / stride + 1,
			(in_size.y - kernel_size + 2*padding) / stride + 1,
			number_filters
		),
		dz_in(
			in_size.b,
			(in_size.x - kernel_size + 2*padding) / stride + 1,
			(in_size.y - kernel_size + 2*padding) / stride + 1,
			number_filters
		),
		padded_in( in_size.b, in_size.x + 2*padding, in_size.y + 2*padding, in_size.z )
	{
		lr = learning_rate / (float)in_size.b;
		this->stride = stride;
		this->kernel_size = kernel_size;
		this->padding = padding;
		this->decay = decay;
		this->momentum = momentum;
		this->number_filters = number_filters;

		assert( (float( in_size.x - kernel_size + 2*padding) / stride + 1)
				==
				((in_size.x - kernel_size + 2*padding) / stride + 1) );

		assert( (float( in_size.y - kernel_size + 2*padding) / stride + 1)
				==
				((in_size.y - kernel_size + 2*padding) / stride + 1) );

		filter_size = 1 * kernel_size * kernel_size * in_size.z;
		for ( int a = 0; a < number_filters; a++ ){
			TensorObject<float> kernel( 1, kernel_size, kernel_size, in_size.z );
			int maxval = filter_size;

			for ( int i = 0; i < kernel_size; ++i ){
				for ( int j = 0; j < kernel_size; ++j ){
					for ( int z = 0; z < in_size.z; ++z ){
						kernel( 0, i, j, z ) = 1.0f / maxval * rand() / float( RAND_MAX );
					}
				}
			}
			filters.push_back( kernel );
		}

		for ( int a = 0; a < number_filters; a++ ){
			TensorObject<GradientObject> filter_grad( 1, kernel_size, kernel_size, in_size.z );
			filter_grads.push_back( filter_grad );
		}

		memset( padded_in.data, 0, padded_in.size.b * padded_in.size.x * padded_in.size.y * padded_in.size.z );

#ifdef GPU_CUDA

		int data_size = in_size.b * in_size.x * in_size.y * in_size.z;
		gpu_dz = gpu_cuda::cudaMakeArray( dz.data, data_size );
		gpu_in = gpu_cuda::cudaMakeArray( in.data, data_size );

		int dz_in_size = in_size.b *
		( (in_size.x - kernel_size + 2*padding) / stride + 1 ) *
		( (in_size.y - kernel_size + 2*padding) / stride + 1 ) *
		number_filters;

		gpu_out = gpu_cuda::cudaMakeArray( out.data, dz_in_size );
		gpu_dz_in = gpu_cuda::cudaMakeArray( dz_in.data, dz_in_size );

		int padded_in_size = in_size.b * (in_size.x + 2*padding) * (in_size.y + 2*padding) * in_size.z;
		gpu_padded_in = gpu_cuda::cudaMakeArray( padded_in.data, padded_in_size );

		gpu_cuda::cudaMakeRandomArray( NULL, filter_size * number_filters, filter_size );
		gpu_filter_grads = gpu_cuda::cudaMakeArray( NULL, filter_size * 2 * number_filters );

#endif

	}

	TensorCoordinate map_to_input( TensorCoordinate out, int z )
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	struct tensor_range_t
	{
		int min_x, min_y;
		int max_x, max_y;
	};

	int normalize_range_min( float f, int max )
	{
		if( f <= 0 ){
			return 0;
		}
		max -= 1;
		if( f >= max ){
			return max;
		}
		return ceil( f );
	}

	int normalize_range_max( float f, int max )
	{
		max -= 1;
		if( f >= max ){
			return max;
		}
		return floor( f );
	}

	tensor_range_t map_to_output( int x, int y )
	{
		float a = x;
		float b = y;
		float stride_inv = 1.0/stride;
		return
		{
			normalize_range_min( (a - kernel_size + 1) * stride_inv, out.size.x ),
			normalize_range_min( (b - kernel_size + 1) * stride_inv, out.size.y ),
			normalize_range_max( a * stride_inv, out.size.x ),
			normalize_range_max( b * stride_inv, out.size.y )
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
		gpu_cuda::convolutionForwardGPU( gpu_in, gpu_out, gpu_padded_in, gpu_filters, in.size.b, in.size.x, in.size.y, in.size.z, out.size.x, out.size.y, out.size.z, padded_in.size.x, padded_in.size.y, padded_in.size.z, padding, kernel_size, stride, filter_size );
	}

	void updateWeightsGPU()
	{
		gpu_cuda::convolutionUpdateWeightsGPU( gpu_filters, gpu_filter_grads, in.size.z, number_filters, kernel_size, momentum, decay, lr);

	}

	void backwardGPU( float *dz_next_layer )
	{
		gpu_cuda::cudaClearArray( gpu_filter_grads, filter_size * 2 * number_filters );
		gpu_cuda::convolutionBackwardGPU( dz_next_layer, gpu_dz_in, gpu_dz, gpu_padded_in, gpu_filters, gpu_filter_grads, dz.size.b, dz.size.x, dz.size.y, dz.size.z, dz_in.size.x, dz_in.size.y, dz_in.size.z, padded_in.size.x, padded_in.size.y, padded_in.size.z, padding, kernel_size, stride, number_filters, filter_size );
	}

#else
	// CPU

	void forward( TensorObject<float>& in,  ThreadPool& thread_pool )
	{
		this->in = in;
		forward( thread_pool );
	}

	void forward( ThreadPool& thread_pool )
	{
		std::vector< std::future<int> > results;

		for ( int b = 0; b < in.size.b; ++b ){
			for ( int z = 0; z < in.size.z; ++z ){
				for ( int y = 0; y < in.size.y; ++y ){
					for ( int x = 0; x < in.size.x; ++x ){
						padded_in( b, padding+x, padding+y, z ) = in( b, x, y, z );
					}
				}
			}

			int filters_size = filters.size();
			for ( int filter = 0; filter < filters_size; ++filter ){
				results.emplace_back( thread_pool.enqueue([&, filter] {

					TensorObject<float> filter_data = filters[filter];

					for ( int y = 0; y < out.size.y; ++y ){
						for ( int x = 0; x < out.size.x; ++x ){
							TensorCoordinate mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
							float sum = 0;
							for ( int z = 0; z < in.size.z; ++z ){
								for ( int j = 0; j < kernel_size; ++j ){
									for ( int i = 0; i < kernel_size; ++i ){
										sum += filter_data( 0, i, j, z ) * padded_in( b, mapped.x + i, mapped.y + j, z );
									}
								}
							}
							out( b, x, y, filter ) = sum;
						}
					}
					return 0;

				}));

			}

			for(auto && result: results){
				result.get();
			}
			results.erase(results.begin(), results.end());
		}
	}

	void updateWeights()
	{
		int filters_size = filters.size();
		for ( int a = 0; a < filters_size; ++a ){
			for ( int z = 0; z < in.size.z; ++z ){
				for ( int j = 0; j < kernel_size; ++j ){
					for ( int i = 0; i < kernel_size; ++i ){
						GradientObject& grad = filter_grads[a].get( 0, i, j, z );
						float m = (grad.grad + grad.grad_prev * momentum);
						grad.grad_prev = m;
						float& w = filters[a].get( 0, i, j, z );
						w -= lr * ( m + (decay * w));
					}
				}
			}
		}
	}

	void backward( TensorObject<float>& dz_next_layer, ThreadPool& thread_pool )
	{
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		int k_end = filter_grads.size();
		int kernel_size_2 = kernel_size * kernel_size;
		int i_end = kernel_size_2 * in.size.z;
		for ( int k = 0; k < k_end; ++k ){
			for ( int i = 0; i < i_end ; ++i ){
					filter_grads[k].data[i].grad = 0;
			}
		}

		int z_max = (int)filters.size();
		std::vector< std::future<int> > thread_results;

		for ( int b = 0; b < in.size.b; ++b ){

			thread_results.emplace_back(thread_pool.enqueue([&, b] {

				// code optimization.
				int dz_in_xy = dz_in.size.y * dz_in.size.x;
				int b_dz_in_xyz = b * dz_in.size.z * dz_in_xy;
				int padded_in_xy = padded_in.size.y * padded_in.size.x;
				int b_padded_in_xyz = b * padded_in.size.z * padded_in_xy;

				int y_end = padded_in.size.y - padding;
				for ( int y = 0; y < y_end; ++y ){

					int x_end = padded_in.size.x - padding;
					for ( int x = 0; x < x_end; ++x ){

						tensor_range_t rn = map_to_output( x, y );

						for ( int z = 0; z < in.size.z; ++z ){

							float sum = 0;
							// float padded_in_value = padded_in( b, x, y, z );
							float padded_in_value = padded_in.data[( b_padded_in_xyz ) + (z * padded_in_xy) + (y * padded_in.size.x) + x];

							for ( int j = rn.min_y; j <= rn.max_y; ++j ){
								int y_miny = y - j * stride;

								for ( int i = rn.min_x; i <= rn.max_x; ++i ){
									int x_minx = x - i * stride;

									int xyz = z * kernel_size_2 + y_miny * kernel_size + x_minx; // ( 0, x_minx, y_miny, z )

									for ( int k = 0; k < z_max; ++k ){
										// float d = dz_in( b, i, j, k );
										float d = dz_in.data[ b_dz_in_xyz + (k * dz_in_xy) + (j * dz_in.size.x) + i];
										// sum += filters[k].get( 0, x_minx, y_miny, z ) * d;
										sum += filters[k].data[xyz] * d;
										// filter_grads[k].get( 0, x_minx, y_miny, z ).grad += padded_in_value * d;
										filter_grads[k].data[xyz].grad += padded_in_value * d;
									}
								}
							}

							if( x>=padding && y>=padding ){
								dz( b, x - padding, y - padding, z ) += sum;
							}
						}

					}
				}

				return 0;
			}));

		}

		for(auto && result: thread_results){
			result.get();
		}
		thread_results.erase(thread_results.begin(), thread_results.end());

	}

#endif

	void saveWeights( ofstream& fout )
	{
		int total_size = 0;
		for ( int a = 0; a < out.size.z; ++a ){
			for ( int i = 0; i <  kernel_size * kernel_size * in.size.z ; ++i ){
				GradientObject grad = filter_grads[a].data[i];
				fout.write(( char * ) &(grad.grad_prev), sizeof( float ) );
				total_size += sizeof(float);
			}
		}

		for ( int a = 0; a < out.size.z; ++a ){
			int size = kernel_size * kernel_size * in.size.z * sizeof( float );
			fout.write(( char * )(filters[a].data), size );
			total_size += size;
		}
		// cout << "- LayerConvolution       : " << to_string(total_size) << " bytes." << endl;
	}

	void loadWeights( ifstream& fin )
	{
		int total_size = 0;
		for ( int a = 0; a < out.size.z; ++a ){
			for ( int i = 0; i <  kernel_size * kernel_size * in.size.z ; ++i ){
				GradientObject& grad = filter_grads[a].data[i];
				fin.read(( char * ) &(grad.grad_prev), sizeof( float ) );
				total_size += sizeof(float);
			}
		}
		for ( int a = 0; a < out.size.z; ++a ){
			int size = kernel_size * kernel_size * in.size.z * sizeof( float );
			fin.read(( char * )(filters[a].data), size );
			total_size += size;
		}
		cout << "- LayerConvolution       : " << to_string(total_size) << " bytes read." << endl;
	}
};
#pragma pack(pop)
