#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerDropout
{
	LayerType type = LayerType::dropout;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<bool> hitmap;
	float p_activation;

	LayerDropout( TensorSize in_size, float p_activation )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		hitmap( in_size.b, in_size.x, in_size.y, in_size.z ),
		p_activation( p_activation )
	{

	}

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for ( int i = 0; i < in.size.b*in.size.x*in.size.y*in.size.z; i++ )
		{
			bool active = (rand() % RAND_MAX) / float( RAND_MAX ) <= p_activation;
			hitmap.data[i] = active;
			out.data[i] = active ? in.data[i] : 0.0f;
		}
	}


	void update_weights()
	{

	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		for ( int i = 0; i < in.size.b*in.size.x*in.size.y*in.size.z; i++ )
			dz.data[i] = hitmap.data[i] ? dz_next_layer.data[i] : 0.0f;
	}
};
#pragma pack(pop)
