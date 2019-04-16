#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerLeakyReLU
{
	LayerType type = LayerType::leaky_relu;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;
	unsigned data_size;

	LayerLeakyReLU( TensorSize in_size
	)
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		data_size = in_size.b * in_size.x * in_size.y * in_size.z;
	}

		void forward(
			TensorObject<float>& in
		)
	{
		this->in = in;
		forward(
		);
	}

	void forward(
	)
	{
		for( int i = 0; i < data_size; ++i ){
			float v = in.data[i];
			if ( v < 0 ){
				v = 0.1 * v;
			}
			out.data[i] = v;
		}

	}

	void update_weights()
	{
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		for( int i = 0; i < data_size; ++i ){
			dz.data[i] +=  (in.data[i] < 0) ? (0.1) : (1.0 * dz_in.data[i]);
		}
	}
};
#pragma pack(pop)
