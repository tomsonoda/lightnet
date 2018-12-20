#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerReLU
{
	LayerType type = LayerType::relu;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;
	unsigned data_size;

	LayerReLU( TensorSize in_size )
		:
		grads_in( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		data_size = in_size.b * in_size.x * in_size.y * in_size.z;
	}

	void activate( TensorObject<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()
	{
		for( int i = 0; i < data_size; i++ ){
			float v = in.data[i];
			if ( v < 0 ){
				v = 0;
			}
			out.data[i] = v;
		}
	}

	void fix_weights()
	{
	}

	void calc_grads( TensorObject<float>& grad_next_layer )
	{
		for( int i = 0; i < data_size; i++ ){
			grads_in.data[i] =  (in.data[i] < 0) ? (0) : (1.0 * grad_next_layer.data[i]);
		}
	}
};
#pragma pack(pop)
