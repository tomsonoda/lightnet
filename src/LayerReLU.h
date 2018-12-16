#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerReLU
{
	LayerType type = LayerType::relu;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;

	LayerReLU( tdsize in_size )
		:
		grads_in( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z )
	{
	}


	void activate( TensorObject<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()
	{
		for ( int i = 0; i < in.size.x; i++ ){
			for ( int j = 0; j < in.size.y; j++ ){
				for ( int z = 0; z < in.size.z; z++ ){
					float v = in( 0, i, j, z );
					if ( v < 0 ){
						v = 0;
					}
					out( 0, i, j, z ) = v;
				}
			}
		}
	}

	void fix_weights()
	{
	}

	void calc_grads( TensorObject<float>& grad_next_layer )
	{
		for ( int i = 0; i < in.size.x; i++ ){
			for ( int j = 0; j < in.size.y; j++ ){
				for ( int z = 0; z < in.size.z; z++ ){
					grads_in( 0, i, j, z ) = (in( 0, i, j, z ) < 0) ? (0) : (1 * grad_next_layer( 0, i, j, z ));
				}
			}
		}
	}
};
#pragma pack(pop)
