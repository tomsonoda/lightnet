#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerSigmoid
{
	LayerType type = LayerType::sigmoid;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;

	LayerSigmoid( TensorSize in_size )
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

	float activator_function( float x )
	{
		//return tanhf( x );
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig;
	}

	float activator_derivative( float x )
	{
		//float t = tanhf( x );
		//return 1 - t * t;
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig * (1 - sig);
	}

	void activate()
	{
		for ( int b = 0; b < in.size.b; b++ ){
			for ( int i = 0; i < in.size.x; i++ ){
				// for ( int j = 0; j < in.size.y; j++ ){
				// 	for ( int z = 0; z < in.size.z; z++ ){
						out( b, i, 0, 0 ) = activator_function(in( b, i, 0, 0 ));
						// out( b, i, 0, 0 ) = in( b, i, 0, 0 );
					// }
			// 	}
			}
		}
	}

	void fix_weights()
	{
	}

	void calc_grads( TensorObject<float>& grad_next_layer )
	{
		for ( int b = 0; b < in.size.b; b++ ){
			for ( int i = 0; i < in.size.x; i++ ){
				// for ( int j = 0; j < in.size.y; j++ ){
				// 	for ( int z = 0; z < in.size.z; z++ ){
						// float sig = 1.0f / (1.0f + exp( - in( b, i, j, z ) ));
						// grads_in( b, i, j, z ) =  (sig * (1-sig)) * grad_next_layer( b, i, j, z );
						grads_in( b, i, 0, 0 ) = activator_derivative( in( b, i, 0, 0 ) ) * grad_next_layer( b, i, 0, 0 );
						// grads_in( b, i, 0, 0 ) = grad_next_layer( b, i, 0, 0 );
				// 	}
				// }
			}
		}
	}
};
#pragma pack(pop)
