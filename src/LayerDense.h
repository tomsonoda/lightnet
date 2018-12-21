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
	TensorObject<float> dW;
	unsigned weigts_data_num;
	unsigned dw_data_size;
	std::vector<GradientObject> gradients;
	float lr;
	float WEIGHT_DECAY;
	float MOMENTUM;
	std::vector<float> input;

	LayerDense( TensorSize in_size, int out_size, float learning_rate, float decay, float momentum)
		:
		grads_in( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, out_size, 1, 1 ),
		weights( 1, in_size.x*in_size.y*in_size.z, out_size, 1 ),
		dW( 1, in_size.x*in_size.y*in_size.z, out_size, 1 )
	{
		gradients = std::vector<GradientObject>( out_size * in_size.b );
		lr = learning_rate / (float)in_size.b;
		int maxval = in_size.x * in_size.y * in_size.z;
		WEIGHT_DECAY = decay;
		MOMENTUM = momentum;
		input = std::vector<float>( out_size );
		weigts_data_num = in_size.x*in_size.y*in_size.z * out_size;
		dw_data_size = in_size.x * in_size.y * in_size.z * out_size * sizeof( float );

		for(int i=0; i<out_size; i++){
			for(int h=0; h<in_size.x*in_size.y*in_size.z; h++){
				// weights( 0, h, i, 0 ) = 0.05 * rand() / float( RAND_MAX );
				weights( 0, h, i, 0 ) = 2.19722f / maxval * rand() / float( RAND_MAX );
			}
		}
	}

	int map( TensorCoordinate d )
	{
		return (d.z * (in.size.x * in.size.y)) + (d.y * (in.size.x)) + d.x;
	}

	void activate( TensorObject<float>& in )
	{
		this->in = in;
		activate();
	}

	float update_weight( float w, float grad, float oldgrad, float multp = 1 )
	{
		float m = (grad + oldgrad * MOMENTUM);
		w -= lr  * m * multp + lr * WEIGHT_DECAY * w;
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
				float sum = 0;
				for ( int i = 0; i < in.size.x; i++ ){
					for ( int j = 0; j < in.size.y; j++ ){
						for ( int z = 0; z < in.size.z; z++ ){
							int m = map( { 0, i, j, z } );
							sum += in( b, i, j, z ) * weights( 0, m, n, 0 );
						}
					}
				}
				// out( b, n, 0, 0 ) = activator_function( inputv );
				out( b, n, 0, 0 ) = sum;
			}
		}
	}

	void fix_weights()
	{
		// printf("learning_rate=%f\n", lr);
		for (int i=0; i<weigts_data_num; i++){
			weights.data[i] = weights.data[i] - lr * 	dW.data[i];
		}
		for ( int i = 0; i < out.size.x * in.size.b; i++ ){
				GradientObject& grad = gradients[ i ];
				update_gradient( grad );
		}
	}

	void calc_grads( TensorObject<float>& grad_next_layer )
	{
		memset( grads_in.data, 0, grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) );
		memset( dW.data, 0, dw_data_size );
		for ( int n = 0; n < out.size.x; n++ ){
			// grad.grad = grad_next_layer( b, n, 0, 0 ) * activator_derivative( input[n] );
			for ( int i = 0; i < in.size.x; i++ ){
				for ( int j = 0; j < in.size.y; j++ ){
					for ( int z = 0; z < in.size.z; z++ ){

						int m = map( { 0, i, j, z } );
						for( int b = 0; b < in.size.b; b++ ){
							GradientObject& grad = gradients[ n*in.size.b + b ];
							grad.grad = grad_next_layer( b, n, 0, 0 );
							grads_in( b, i, j, z ) += grad_next_layer( b, n, 0, 0 ) * weights( 0, m, n, 0 );
							dW (0, m, n, 0) += in(b, i, j, z) * (grad.grad + grad.oldgrad * MOMENTUM) + (WEIGHT_DECAY * weights(0, m, n, 0));
						}
					}
				}
			}
		}
	}
};
#pragma pack(pop)
