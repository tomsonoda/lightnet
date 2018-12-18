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
	TensorObject<float> weights;
	TensorObject<float> dW;
	float lr;
	unsigned in_size_xy;
	unsigned dw_data_size;
	unsigned grads_in_data_size;
	unsigned weigts_data_num;

	LayerDense( TensorSize in_size, int out_size, float learning_rate)
		:
		grads_in( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, out_size, 1, 1 ),
		weights( 1, in_size.x*in_size.y*in_size.z, out_size, 1 ),
		// dW( in_size.b, in_size.x*in_size.y*in_size.z, out_size, 1 )
		dW( 1, in_size.x*in_size.y*in_size.z, out_size, 1 )
	{
		lr = learning_rate / (float)in_size.b;
		for(int i=0; i<out_size; i++){
			for(int h=0; h<in_size.x*in_size.y*in_size.z; h++){
				weights(0,h,i,0) = 0.05 * rand() / float( RAND_MAX );
			}
		}
		in_size_xy = in_size.x*in_size.y;
		dw_data_size = dW.size.b *dW.size.x *dW.size.y*dW.size.z * sizeof( float );
		grads_in_data_size = grads_in.size.b *grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) ;
		weigts_data_num = in_size.x*in_size.y*in_size.z * out_size;
	}

	void activate( TensorObject<float>& in )
	{
		this->in = in;
		activate();
	}
/*
	int map( TensorCoordinate d )
	{
		// return (d.z * (in.size.x * in.size.y)) + (d.y * (in.size.x)) + d.x;
		return (d.z * in_size_xy) + (d.y * (in.size.x)) + d.x;
	}
*/
	void activate()
	{
		for ( int n = 0; n < out.size.x; n++ ){
			for ( int b = 0; b < in.size.b; b++ ){
				float sum = 0;
				for (int i = 0; i < in.size.x; i++ ){
					for (int j = 0; j < in.size.y; j++ ){
						for (int z = 0; z < in.size.z; z++ ){
							// int m = map( { 0, i, j, z } );
							int m = (z * in_size_xy) + (j * in.size.x) + i;
							sum += weights( 0, m, n, 0 ) * in( b, i, j, z );
						}
					}
				}
				out( b, n, 0, 0 ) = sum;
			}
		}
	}

	void fix_weights()
	{
		/*
		for (int n=0; n<out.size.x; n++){
			for (int i=0; i<in.size.x; i++ ){
				for (int j=0; j<in.size.y; j++ ){
					for (int z=0; z<in.size.z; z++ ){
						// float dW_sum = 0.0;
						// int m = map( {0, i, j, z } );
						int m = (z * in_size_xy) + (j * in.size.x) + i;
						// for ( int b = 0; b < in.size.b; b++ ){
						// 		dW_sum += dW(b, m, n, 0);
						// }
						// weights(0, m, n, 0) = weights(0, m, n, 0) - lr * dW_sum;
						weights(0, m, n, 0) = weights(0, m, n, 0) - lr * 	dW (0, m, n, 0);
					}
				}
			}
		}
		*/
		for (int i=0; i<weigts_data_num; i++){
			weights.data[i] = weights.data[i] - lr * 	dW.data[i];
		}
	}

	void calc_grads( TensorObject<float>& grad_next_layer ){
		memset( dW.data, 0, dw_data_size );
		memset( grads_in.data, 0, grads_in_data_size);
		for (int n=0; n<out.size.x; n++){
			for ( int b = 0; b < in.size.b; b++ ){
				for (int i=0; i<in.size.x; i++ ){
					for (int j=0; j<in.size.y; j++ ){
						for (int z=0; z<in.size.z; z++ ){
							// int m = map( {0, i, j, z } );
							int m = (z * in_size_xy) + (j * in.size.x) + i;
							// dW (m, n, 0) = in(i, j, z) * grad_next_layer(n, 0, 0);
							// grads_in(i, 0, 0) += weights(m, n, 0) * grad_next_layer(n, 0, 0) * activator_derivative(z_[n]);
							float g = grad_next_layer(b, n, 0, 0);
							// dW (b, m, n, 0) = in(b, i, j, z) * g;
							dW (0, m, n, 0) += in(b, i, j, z) * g;
							grads_in(b, i, 0, 0) += weights(0, m, n, 0) * g;
						}
					}
				}
			}
		}
	}
};
#pragma pack(pop)
