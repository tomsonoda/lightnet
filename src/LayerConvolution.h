#pragma once
#include "LayerObject.h"
#include "GradientObject.h"

#pragma pack(push, 1)
struct LayerConvolution
{
	LayerType type = LayerType::conv;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;
	TensorObject<float> padded_in;
	std::vector<TensorObject<float>> filters;
	std::vector<TensorObject<GradientObject>> filter_grads;
	uint16_t stride;
	uint16_t kernel_size;
	uint16_t padding;
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

		assert( (float( in_size.x - kernel_size + 2*padding) / stride + 1)
				==
				((in_size.x - kernel_size + 2*padding) / stride + 1) );

		assert( (float( in_size.y - kernel_size + 2*padding) / stride + 1)
				==
				((in_size.y - kernel_size + 2*padding) / stride + 1) );

		for ( int a = 0; a < number_filters; a++ ){
			TensorObject<float> kernel( 1, kernel_size, kernel_size, in_size.z );
			int maxval = kernel_size * kernel_size * in_size.z;

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

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for ( int b = 0; b < in.size.b; ++b ){
			for ( int x = 0; x < in.size.x; ++x ){
				for ( int y = 0; y < in.size.y; ++y ){
					for ( int z = 0; z < in.size.z; ++z ){
						padded_in( b, padding+x, padding+y, z ) = in( b, x, y, z );
					}
				}
			}

			int filters_size = filters.size();
			for ( int filter = 0; filter < filters_size; ++filter ){
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
			}
		}
	}

	void update_weights()
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

	void backward( TensorObject<float>& dz_next_layer )
	{
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		int k_end = filter_grads.size();
		for ( int k = 0; k < k_end; ++k ){
			for ( int i = 0; i < kernel_size * kernel_size * in.size.z; ++i ){
					filter_grads[k].data[i].grad = 0;
			}
		}

		int z_max = (int)filters.size();

		for ( int b = 0; b < in.size.b; ++b ){
			for ( int x = 0; x < padded_in.size.x; ++x ){
				for ( int y = 0; y < padded_in.size.y; ++y ){

					tensor_range_t rn = map_to_output( x, y );

					for ( int z = 0; z < in.size.z; ++z ){
						float sum_error = 0;
						float padded_in_value = padded_in( b, x, y, z );

						for ( int j = rn.min_y; j <= rn.max_y; ++j ){
							int y_miny = y - j * stride;

							for ( int i = rn.min_x; i <= rn.max_x; ++i ){
								int x_minx = x - i * stride;

								for ( int k = 0; k < z_max; ++k ){
									float d = dz_in( b, i, j, k );
									sum_error += filters[k].get( 0, x_minx, y_miny, z ) * d;
									filter_grads[k].get( 0, x_minx, y_miny, z ).grad += padded_in_value * d;
								}
							}
						}

						float x_padding = x - padding;
						float y_padding = y - padding;
						if(x>=padding && y>=padding && x_padding<in.size.x && y_padding<in.size.y ){
							dz( b, x_padding, y_padding, z ) += sum_error;
						}
					}

				}
			}
		}

	}
};
#pragma pack(pop)
