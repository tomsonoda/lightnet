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

			for ( int i = 0; i < kernel_size; i++ ){
				for ( int j = 0; j < kernel_size; j++ ){
					for ( int z = 0; z < in_size.z; z++ ){
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
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
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
		return
		{
			normalize_range_min( (a - kernel_size + 1) / stride, out.size.x ),
			normalize_range_min( (b - kernel_size + 1) / stride, out.size.y ),
			0,
			normalize_range_max( a / stride, out.size.x ),
			normalize_range_max( b / stride, out.size.y ),
			(int)filters.size() - 1,
		};
	}

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for ( int b = 0; b < in.size.b; b++ ){
			for ( int x = 0; x < in.size.x; x++ ){
				for ( int y = 0; y < in.size.y; y++ ){
					for ( int z = 0; z < in.size.z; z++ ){
						padded_in( b, padding+x, padding+y, z ) = in( b, x, y, z );
					}
				}
			}
		}

		for ( int filter = 0; filter < filters.size(); filter++ ){
			TensorObject<float>& filter_data = filters[filter];

			for ( int b = 0; b < out.size.b; b++ ){
				for ( int x = 0; x < out.size.x; x++ ){
					for ( int y = 0; y < out.size.y; y++ ){
						TensorCoordinate mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
						float sum = 0;

						for ( int i = 0; i < kernel_size; i++ ){
							for ( int j = 0; j < kernel_size; j++ ){
								for ( int z = 0; z < in.size.z; z++ ){
									float f = filter_data( 0, i, j, z );
									float v = padded_in( b, mapped.x + i, mapped.y + j, z );
									sum += f*v;
								}
							}
						}
						out( b, x, y, filter ) = sum;
					}
				}
			}
		}
	}

	float update_weight( float w, GradientObject& grad, float multp = 1 )
	{
		float m = (grad.grad + grad.grad_prev * momentum);
		w -= lr  * ( (m * multp) + (decay * w));
		return w;
	}

	void update_gradient( GradientObject& grad )
	{
		grad.grad_prev = (grad.grad + grad.grad_prev * momentum);
	}

	void update_weights()
	{
		for ( int a = 0; a < filters.size(); a++ ){
			for ( int i = 0; i < kernel_size; i++ ){
				for ( int j = 0; j < kernel_size; j++ ){
					for ( int z = 0; z < in.size.z; z++ ){
						float& w = filters[a].get( 0, i, j, z );
						GradientObject& grad = filter_grads[a].get( 0, i, j, z );

						float m = (grad.grad + grad.grad_prev * momentum);
						w -= lr * ( m + (decay * w));
						grad.grad_prev = m;

						// w = update_weight( w, grad );
						// update_gradient( grad );
					}
				}
			}
		}
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; i++ ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		for ( int k = 0; k < filter_grads.size(); k++ ){
			for ( int i = 0; i < kernel_size; i++ ){
				for ( int j = 0; j < kernel_size; j++ ){
					for ( int z = 0; z < in.size.z; z++ ){
						filter_grads[k].get( 0, i, j, z ).grad = 0;
					}
				}
			}
		}

		for ( int b = 0; b < in.size.b; b++ ){
			for ( int x = 0; x < padded_in.size.x; x++ ){
				for ( int y = 0; y < padded_in.size.y; y++ ){
					tensor_range_t rn = map_to_output( x, y );
					for ( int z = 0; z < in.size.z; z++ ){

						float sum_error = 0;
						for ( int i = rn.min_x; i <= rn.max_x; i++ ){
							int minx = i * stride;
							for ( int j = rn.min_y; j <= rn.max_y; j++ ){
								int miny = j * stride;
								int x_minx = x - minx;
								int y_miny = y - miny;
								for ( int k = rn.min_z; k <= rn.max_z; k++ ){
									float d = dz_in( b, i, j, k );
									sum_error += filters[k].get( 0, x_minx, y_miny, z ) * d;
									filter_grads[k].get( 0, x_minx, y_miny, z ).grad += padded_in( b, x, y, z ) * d;
								}
							}
						}

						if(x>=padding && y>=padding && x-padding<in.size.x && y-padding<in.size.y ){
							dz( b, x-padding, y-padding, z ) += sum_error;
						}
					}

				}
			}
		}

	}
};
#pragma pack(pop)
