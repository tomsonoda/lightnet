#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "LayerObject.h"
#include "GradientObject.h"

#pragma pack(push, 1)
struct LayerDense
{
	LayerType type = LayerType::dense;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> weights;
	// TensorObject<float> dW;
	std::vector<GradientObject> gradients;

	float lr;
	// unsigned in_size_xy;
	// unsigned dw_data_size;
	unsigned grads_in_data_size;
	unsigned weigts_data_num;
	unsigned WEIGHT_DECAY;
	unsigned MOMENTUM;
	std::vector<float> input;

	LayerDense( TensorSize in_size, int out_size, float learning_rate, float decay, float momentum)
		:
		grads_in( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, out_size, 1, 1 ),
		weights( 1, in_size.x*in_size.y*in_size.z, out_size, 1 )
		// dW( 1, in_size.x*in_size.y*in_size.z, out_size, 1 )
	{
		gradients = std::vector<GradientObject>( out_size * in_size.b );
		lr = learning_rate / (float)in_size.b;
		int maxval = in_size.x * in_size.y * in_size.z;
		WEIGHT_DECAY = decay;
		MOMENTUM = momentum;
		input = std::vector<float>( out_size );

		for(int i=0; i<out_size; i++){
			for(int h=0; h<in_size.x*in_size.y*in_size.z; h++){
				// weights( 0, h, i, 0 ) = 0.05 * rand() / float( RAND_MAX );
				weights( 0, h, i, 0 ) = 2.19722f / maxval * rand() / float( RAND_MAX );
			}
		}
		// in_size_xy = in_size.x*in_size.y;
		// dw_data_size = dW.size.b *dW.size.x *dW.size.y*dW.size.z * sizeof( float );
		grads_in_data_size = grads_in.size.b *grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) ;
		weigts_data_num = in_size.x*in_size.y*in_size.z * out_size;
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

	int map( TensorCoordinate d )
	{
		return (d.z * (in.size.x * in.size.y)) + (d.y * (in.size.x)) + d.x;
		// return (d.z * in_size_xy) + (d.y * (in.size.x)) + d.x;
	}

	void activate( TensorObject<float>& in )
	{
		this->in = in;
		activate();
	}

	float update_weight( float w, GradientObject& grad, float multp = 1 )
	{
		float m = (grad.grad + grad.oldgrad * MOMENTUM);
		w -= lr  * m * multp + lr * WEIGHT_DECAY * w;
		// float m = (grad.grad + grad.oldgrad * MOMENTUM);
		// w -= lr  * ( (m * multp) + (WEIGHT_DECAY * w));
		return w;
	}

	void update_gradient( GradientObject& grad )
	{
		grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
	}


	void activate()
	{
		for ( int b = 0; b < in.size.b; b++ ){
			for ( int n = 0; n < out.size.x; n++ ){
				float inputv = 0;
				for ( int i = 0; i < in.size.x; i++ ){
					for ( int j = 0; j < in.size.y; j++ ){
						for ( int z = 0; z < in.size.z; z++ ){
							int m = map( { 0, i, j, z } );
							inputv += in( b, i, j, z ) * weights( 0, m, n, 0 );
						}
					}
				}
				input[n] = inputv;
				// out( b, n, 0, 0 ) = activator_function( inputv );
				out( b, n, 0, 0 ) = inputv;
			}
		}
	}

	void fix_weights()
	{
		for ( int b = 0; b < in.size.b; b++ ){
			for ( int n = 0; n < out.size.x; n++ ){
				GradientObject& grad = gradients[n];
				for ( int i = 0; i < in.size.x; i++ ){
					for ( int j = 0; j < in.size.y; j++ ){
						for ( int z = 0; z < in.size.z; z++ ){
							int m = map( { 0, i, j, z } );
							float& w = weights( 0, m, n, 0 );
							w = update_weight( w, grad, in( b, i, j, z ) );
						}
					}
				}
				update_gradient( grad );
			}
		}
	}

	void calc_grads( TensorObject<float>& grad_next_layer )
	{
		memset( grads_in.data, 0, grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) );
		for ( int b = 0; b < in.size.b; b++ ){
			for ( int n = 0; n < out.size.x; n++ ){
				GradientObject& grad = gradients[n];
				// grad.grad = grad_next_layer( b, n, 0, 0 ) * activator_derivative( input[n] );
				grad.grad = grad_next_layer( b, n, 0, 0 );

				for ( int i = 0; i < in.size.x; i++ ){
					for ( int j = 0; j < in.size.y; j++ ){
						for ( int z = 0; z < in.size.z; z++ ){
							int m = map( { 0, i, j, z } );
							grads_in( b, i, j, z ) += grad.grad * weights( 0, m, n, 0 );
						}
					}
				}
			}
		}
	}

};
#pragma pack(pop)
