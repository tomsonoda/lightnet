#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerSoftmax
{
	LayerType type = LayerType::softmax;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;

	LayerSoftmax( tdsize in_size )
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
		float max_v = 0.0;
		for ( int i = 0; i < in.size.x; i++ ){
			for ( int j = 0; j < in.size.y; j++ ){
				for ( int z = 0; z < in.size.z; z++ ){
					float v = in( i, j, z );
					if(v>max_v){
						max_v = v;
					}
				}
			}
		}

		float sum_of_elems = 0.0;
		for ( int i = 0; i < in.size.x; i++ ){
			for ( int j = 0; j < in.size.y; j++ ){
				for ( int z = 0; z < in.size.z; z++ ){
					float v = in( i, j, z );
					out( i, j, z ) = exp(v + max_v);
					sum_of_elems = sum_of_elems + out( i, j, z );
				}
			}
		}

		for ( int i = 0; i < in.size.x; i++ ){
			for ( int j = 0; j < in.size.y; j++ ){
				for ( int z = 0; z < in.size.z; z++ ){
					out( i, j, z ) = out( i, j, z ) / sum_of_elems;
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
					grads_in( i, j, z ) = grad_next_layer( i, j, z );
				}
			}
		}
	}
};
#pragma pack(pop)
