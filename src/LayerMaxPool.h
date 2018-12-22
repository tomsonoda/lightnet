#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerPool
{
	LayerType type = LayerType::max_pool;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	uint16_t stride;
	uint16_t extend_filter;

	LayerPool( uint16_t stride, uint16_t extend_filter, TensorSize in_size )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out(
			in_size.b,
			(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			in_size.z
		)

	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );
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
		return
		{
			normalize_range( (a - extend_filter + 1) / stride, out.size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)out.size.z - 1,
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
			for ( int x = 0; x < out.size.x; x++ ){
				for ( int y = 0; y < out.size.y; y++ ){
					for ( int z = 0; z < out.size.z; z++ ){
						TensorCoordinate mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
						float mval = -FLT_MAX;
						for ( int i = 0; i < extend_filter; i++ ){
							for ( int j = 0; j < extend_filter; j++ ){
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

	void update_weights()
	{
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		for ( int b = 0; b < in.size.b; b++ ){

			for ( int x = 0; x < in.size.x; x++ ){
				for ( int y = 0; y < in.size.y; y++ ){
					range_t rn = map_to_output( x, y );
					for ( int z = 0; z < in.size.z; z++ ){
						float sum_error = 0;
						for ( int i = rn.min_x; i <= rn.max_x; i++ ){
							// int minx = i * stride;
							for ( int j = rn.min_y; j <= rn.max_y; j++ ){
								// int miny = j * stride;
								int is_max = in( b, x, y, z ) == out( b, i, j, z ) ? 1 : 0;
								sum_error += is_max * dz_next_layer( b, i, j, z );
							}
						}
						dz( b, x, y, z ) = sum_error;
					}
				}
			}
		}

	}
};
#pragma pack(pop)
