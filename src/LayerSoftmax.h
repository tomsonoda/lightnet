#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerSoftmax
{
	LayerType type = LayerType::softmax;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	LayerSoftmax( TensorSize in_size )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z )
	{
	}

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for ( int b = 0; b < in.size.b; ++b ){

			float max_v = 0.0;
			for ( int i = 0; i < in.size.x; ++i ){
				float v = in( b, i, 0, 0 );
				if(v>max_v){
					max_v = v;
				}
			}

			float sum = 0.0;
			for ( int i = 0; i < in.size.x; ++i ){
				float v = in( b, i, 0, 0 );
				v = exp(v - max_v);
				out( b, i, 0, 0 ) = v;
				sum += v;
			}

			for ( int i = 0; i < in.size.x; ++i ){
				out( b, i, 0, 0 ) = out( b, i, 0, 0 ) / sum;
			}

		}
	}

	void updateWeights()
	{
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		for ( int i = 0; i < in.size.b * in.size.x * in.size.y * in.size.z; ++i ){
			dz.data[i] += dz_in.data[i];
		}
	}
};
#pragma pack(pop)
