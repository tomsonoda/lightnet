#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerFullyConnected
{
	LayerType type = LayerType::fc;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;
	std::vector<float> input;
	std::vector<float> z_;
	TensorObject<float> weights;
	TensorObject<float> dW;
	ActivationType a_type;

	LayerFullyConnected( tdsize in_size, int out_size, ActivationType at)
		:
		grads_in( in_size.x, in_size.y, in_size.z ),
		in( in_size.x, in_size.y, in_size.z ),
		out( out_size, 1, 1 ),
		weights( in_size.x*in_size.y*in_size.z, out_size, 1 ),
		dW( in_size.x*in_size.y*in_size.z, out_size, 1 )
	{
		// int maxval = in_size.x * in_size.y * in_size.z;
		input = std::vector<float>( out_size );
		z_ =  std::vector<float>( out_size );
		a_type = at;

		for(int i=0; i<out_size; i++){
			for(int h=0; h<in_size.x*in_size.y*in_size.z; h++){
				weights(h,i,0) = 0.05 * rand() / float( RAND_MAX );
				// weights( h, i, 0 ) = 2.19722f / maxval * rand() / float( RAND_MAX );
				dW(h,i,0) = 0;
			}
		}
	}

	float activator_function( float x )
	{
		if(a_type==ActivationType::sigmoid){
			float sig = 1.0f / (1.0f + exp( -x ));
			return sig;
		}else if(a_type==ActivationType::relu){
			return (x<0?0:x);
		}else{ // softmax
				return x;
		}
	}

	float activator_derivative( float x )
	{
		if(a_type==ActivationType::sigmoid){
			float sig = 1.0f / (1.0f + exp( -x ));
			return sig * (1-sig);
		}else if(a_type==ActivationType::relu){
			return (x<0?0:1);
		}else{ // softmax
			return 1;
		}
	}

	void softmax()
	{
		float max_v = 0.0;
		for (int n=0; n<out.size.x; n++){
			float v = z_[n];
			if(v>max_v){
				max_v = v;
			}
		}

		float sum_of_elems = 0.0;
		for (int n=0; n<out.size.x; n++){
			input[n] = exp(z_[n] - max_v);
			sum_of_elems += input[n];
		}

		for (int n=0; n<out.size.x; n++){
			out( n, 0, 0 ) = input[n] / sum_of_elems;
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
		}
		if(a_type==ActivationType::softmax){
			softmax();
		}else{
			for (int n=0; n<out.size.x; n++){
				out(n, 0, 0) = activator_function(z_[n]);
			}
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
						dW (m, n, 0) = in(i, j, z) * grad_next_layer(n, 0, 0);
						grads_in(i, 0, 0) += weights(m, n, 0) * grad_next_layer(n, 0, 0) * activator_derivative(z_[n]);
					}
				}
			}
		}
	}
};
#pragma pack(pop)
