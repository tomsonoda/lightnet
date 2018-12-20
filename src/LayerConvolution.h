#pragma once
#include "LayerObject.h"
#include "GradientObject.h"

#define LEARNING_RATE 0.01 // 0.01
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

#pragma pack(push, 1)
struct LayerConvolution
{
	LayerType type = LayerType::conv;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;
	std::vector<TensorObject<float>> filters;
	std::vector<TensorObject<GradientObject>> filter_grads;
	uint16_t stride;
	uint16_t extend_filter;
	uint16_t padding;
	float learning_rate;

	LayerConvolution( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, uint16_t padding, TensorSize in_size )
		:
		grads_in( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out(
			in_size.b,
			(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			number_filters
		)

	{
		learning_rate = LEARNING_RATE / (float)in_size.b;

		this->stride = stride;
		this->extend_filter = extend_filter;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );

		for ( int a = 0; a < number_filters; a++ ){
			TensorObject<float> kernel( 1, extend_filter, extend_filter, in_size.z );
			int maxval = extend_filter * extend_filter * in_size.z;
			for ( int i = 0; i < extend_filter; i++ ){
				for ( int j = 0; j < extend_filter; j++ ){
					for ( int z = 0; z < in_size.z; z++ ){
						kernel( 0, i, j, z ) = 1.0f / maxval * rand() / float( RAND_MAX );
					}
				}
			}
			filters.push_back( kernel );
		}

		for ( int i = 0; i < number_filters; i++ ){
			TensorObject<GradientObject> kernel( 1, extend_filter, extend_filter, in_size.z );
			filter_grads.push_back( kernel );
		}

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
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	int normalize_range( float f, int max, bool lim_min )
	{
		if( f <= 0 ){
			return 0;
		}
		max -= 1;
		if( f >= max ){
			return max;
		}

		if( lim_min ){// left side of inequality
			return ceil( f );
		}else{
			return floor( f );
		}
	}

	range_t map_to_output( int x, int y )
	{
		float a = x;
		float b = y;
		return
		{
			normalize_range( (a - extend_filter + 1) / stride, out.size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)filters.size() - 1,
		};
	}

	void activate( TensorObject<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()
	{
		for ( int filter = 0; filter < filters.size(); filter++ ){
			TensorObject<float>& filter_data = filters[filter];

			for ( int b = 0; b < in.size.b; b++ ){
				for ( int x = 0; x < out.size.x; x++ ){
					for ( int y = 0; y < out.size.y; y++ ){
						TensorCoordinate mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
						float sum = 0;

						for ( int i = 0; i < extend_filter; i++ ){
							for ( int j = 0; j < extend_filter; j++ ){
								for ( int z = 0; z < in.size.z; z++ ){
									float f = filter_data( 0, i, j, z );
									float v = in( b, mapped.x + i, mapped.y + j, z );
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
		float m = (grad.grad + grad.oldgrad * MOMENTUM);
		w -= learning_rate  * ( (m * multp) + (WEIGHT_DECAY * w));
		return w;
	}

	void update_gradient( GradientObject& grad )
	{
		grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
	}

	void fix_weights()
	{
		for ( int a = 0; a < filters.size(); a++ ){
			for ( int i = 0; i < extend_filter; i++ ){
				for ( int j = 0; j < extend_filter; j++ ){
					for ( int z = 0; z < in.size.z; z++ ){
						float& w = filters[a].get( 0, i, j, z );
						GradientObject& grad = filter_grads[a].get( 0, i, j, z );
						w = update_weight( w, grad );
						update_gradient( grad );
					}
				}
			}
		}
	}

	void calc_grads( TensorObject<float>& grad_next_layer )
	{
		for ( int b = 0; b < in.size.b; b++ ){

			for ( int k = 0; k < filter_grads.size(); k++ ){
				for ( int i = 0; i < extend_filter; i++ ){
					for ( int j = 0; j < extend_filter; j++ ){
						for ( int z = 0; z < in.size.z; z++ ){
							filter_grads[k].get( 0, i, j, z ).grad = 0;
						}
					}
				}
			}
			for ( int x = 0; x < in.size.x; x++ ){
				for ( int y = 0; y < in.size.y; y++ ){
					range_t rn = map_to_output( x, y );
					for ( int z = 0; z < in.size.z; z++ ){
						float sum_error = 0;
						for ( int i = rn.min_x; i <= rn.max_x; i++ ){
							int minx = i * stride;
							for ( int j = rn.min_y; j <= rn.max_y; j++ ){
								int miny = j * stride;
								for ( int k = rn.min_z; k <= rn.max_z; k++ ){
									int w_applied = filters[k].get( 0, x - minx, y - miny, z );
									sum_error += w_applied * grad_next_layer( b, i, j, k );
									filter_grads[k].get( 0, x - minx, y - miny, z ).grad += in( b, x, y, z ) * grad_next_layer( b, i, j, k );
								}
							}
						}
						grads_in( b, x, y, z ) = sum_error;
					}
				}
			}
		}
	}
};
#pragma pack(pop)
