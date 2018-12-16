#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerSigmoid
{
	LayerType type = LayerType::relu;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;

	LayerSigmoid( tdsize in_size )
		:
		grads_in( in_size.x, in_size.y, in_size.z ),
		in( in_size.x, in_size.y, in_size.z ),
		out( in_size.x, in_size.y, in_size.z )
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
					out( i, j, z ) = 1.0f / (1.0f + exp( - in( i, j, z ) ));
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
					float sig = 1.0f / (1.0f + exp( - in( i, j, z ) ));
					grads_in( i, j, z ) =  (sig * (1-sig)) * grad_next_layer( i, j, z );
				}
			}
		}
	}
};
#pragma pack(pop)
