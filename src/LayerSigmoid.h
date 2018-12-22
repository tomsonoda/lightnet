#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerSigmoid
{
	LayerType type = LayerType::sigmoid;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	unsigned in_total_size;
	LayerSigmoid( TensorSize in_size )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		in_total_size = in_size.b *in_size.x *in_size.y *in_size.z;
	}

	void activate( TensorObject<float>& in )
	{
		this->in = in;
		activate();
	}

	float activator_function( float x )
	{
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig;
	}

	float activator_derivative( float x )
	{
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig * (1 - sig);
	}

	void activate()
	{
		for ( int i = 0; i < in_total_size; i++ ){
			out.data[i] = activator_function(in.data[i]);
		}
	}

	void update_weights()
	{
	}

	void calc_grads( TensorObject<float>& dz_next_layer )
	{
		for ( int i = 0; i < in_total_size; i++ ){
			dz.data[i] = activator_derivative( in.data[i] ) * dz_next_layer.data[i];
		}
	}
};
#pragma pack(pop)
