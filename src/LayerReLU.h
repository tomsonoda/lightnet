#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerReLU
{
	LayerType type = LayerType::relu;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;
	int data_size;

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
		/*
		for ( int b = 0; b < in.size.b; b++ ){
			for ( int i = 0; i < in.size.x; i++ ){
				for ( int j = 0; j < in.size.y; j++ ){
					for ( int z = 0; z < in.size.z; z++ ){
						float v = in( b, i, j, z );
						if ( v < 0 ){
							v = 0;
						}
						out( b, i, j, z ) = v;
					}
				}
			}
		}
		*/
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
		/*
		for ( int b = 0; b < in.size.b; b++ ){
			for ( int i = 0; i < in.size.x; i++ ){
				for ( int j = 0; j < in.size.y; j++ ){
					for ( int z = 0; z < in.size.z; z++ ){
						grads_in( b, i, j, z ) = (in( b, i, j, z ) < 0) ? (0) : (grad_next_layer( b, i, j, z ));
					}
				}
			}
		}
		*/
		for( int i = 0; i < data_size; i++ ){
			grads_in.data[i] =  (in.data[i] < 0) ? (0) : (grad_next_layer.data[i]);
		}
	}
};
#pragma pack(pop)
