#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerDense
{
	LayerType type = LayerType::dense;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;
	std::vector<float> input;
	std::vector<float> z_;
	TensorObject<float> weights;
	TensorObject<float> dW;

	LayerDense( tdsize in_size, int out_size)
		:
		grads_in( in_size.x, in_size.y, in_size.z ),
		in( in_size.x, in_size.y, in_size.z ),
		out( out_size, 1, 1 ),
		weights( in_size.x*in_size.y*in_size.z, out_size, 1 ),
		dW( in_size.x*in_size.y*in_size.z, out_size, 1 )
	{
		input = std::vector<float>( out_size );
		z_ =  std::vector<float>( out_size );

		for(int i=0; i<out_size; i++){
			for(int h=0; h<in_size.x*in_size.y*in_size.z; h++){
				weights(h,i,0) = 0.05 * rand() / float( RAND_MAX );
				dW(h,i,0) = 0;
			}
		}
	}

	void activate( TensorObject<float>& in )
	{
		this->in = in;
		activate();
	}

	int map( PointObject d )
	{
		return (d.z * (in.size.x * in.size.y)) + (d.y * (in.size.x)) + d.x;
	}

	void activate()
	{
		for ( int n = 0; n < out.size.x; n++ ){
			float inputv = 0;
			for (int i=0; i<in.size.x; i++ ){
				for (int j=0; j<in.size.y; j++ ){
					for (int z=0; z<in.size.z; z++ ){
						int m = map( {i, j, z} );;
						inputv += weights(m, n, 0) * in(i,j,z);
					}
				}
			}
			z_[n] = inputv;
			out(n, 0, 0) = inputv;
		}
	}

	void fix_weights()
	{
		for (int n=0; n<out.size.x; n++){
			for (int i=0; i<in.size.x; i++ ){
				for (int j=0; j<in.size.y; j++ ){
					for (int z=0; z<in.size.z; z++ ){
						int m = map( { i, j, z } );
						weights(m, n, 0) = weights(m, n, 0) - 0.001 * dW(m, n, 0);
					}
				}
			}
		}
	}

	void calc_grads( TensorObject<float>& grad_next_layer ){
		memset( grads_in.data, 0, grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) );
		for (int n=0; n<out.size.x; n++){
			for (int i=0; i<in.size.x; i++ ){
				for (int j=0; j<in.size.y; j++ ){
					for (int z=0; z<in.size.z; z++ ){
						int m = map( { i, j, z } );
						// dW (m, n, 0) = in(i, j, z) * grad_next_layer(n, 0, 0);
						// grads_in(i, 0, 0) += weights(m, n, 0) * grad_next_layer(n, 0, 0) * activator_derivative(z_[n]);
						dW (m, n, 0) = in(i, j, z) * grad_next_layer(n, 0, 0);
						grads_in(i, 0, 0) += weights(m, n, 0) * grad_next_layer(n, 0, 0);
					}
				}
			}
		}
	}
};
#pragma pack(pop)
