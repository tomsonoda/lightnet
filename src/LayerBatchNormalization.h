#pragma once
#include "LayerObject.h"
#include <math.h>

#pragma pack(push, 1)
struct LayerBatchNormalization
{
	LayerType type = LayerType::batch_normalization;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> mean;
	TensorObject<float> variance;
	TensorObject<float> inv_variance;
	TensorObject<float> xhat;
	TensorObject<float> dxhat;
	TensorObject<float> gamma;
	TensorObject<float> dgamma;
	TensorObject<float> beta;
	TensorObject<float> dbeta;
	float scale;
	float lr;

	LayerBatchNormalization( TensorSize in_size, float learning_rate )
		:
		grads_in( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		mean( 1, in_size.x, in_size.y, in_size.z ),
		variance(1, in_size.x, in_size.y, in_size.z ),
		inv_variance(1, in_size.x, in_size.y, in_size.z ),
		xhat( in_size.b, in_size.x, in_size.y, in_size.z ),
		dxhat( in_size.b, in_size.x, in_size.y, in_size.z ),
		gamma( 1, in_size.x, in_size.y, in_size.z  ),
		dgamma( 1, in_size.x, in_size.y, in_size.z ),
		beta( 1, in_size.x, in_size.y, in_size.z ),
		dbeta( 1, in_size.x, in_size.y, in_size.z )
	{
		lr = learning_rate / (float)in_size.b;
		for(int i=0; i<in_size.x*in_size.y*in_size.z; i++){
			gamma.data[i] = 0.05 * rand() / float( RAND_MAX );
			beta.data[i]  = 0.05 * rand() / float( RAND_MAX );
		}
	}

	void activate( TensorObject<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()
	{
		scale = 1.0f / (float)in.size.b;

		for ( int i = 0; i < in.size.x; i++ ){
			for ( int j = 0; j < in.size.y; j++ ){
				for ( int z = 0; z < in.size.z; z++ ){
					float sum = 0;
					for ( int b = 0; b < in.size.b; b++ ){
							sum += in(b, i, j, z);
					}
					mean(0, i, j, z) = sum * scale;
				}
			}
		}

		for ( int i = 0; i < in.size.x; i++ ){
			for ( int j = 0; j < in.size.y; j++ ){
				for ( int z = 0; z < in.size.z; z++ ){
					float sum = 0;
					for ( int b = 0; b < in.size.b; b++ ){
						sum += pow( (in(b, i, j, z) - mean(0, i, j, z)), 2);
					}
					variance(0, i, j, z) = sum * scale;

					float inv_v = 1.0f / sqrt(variance( 0, i, j, z )+0.00001f);
					inv_variance(0, i, j, z ) = inv_v;
					for ( int b = 0; b < in.size.b; b++ ){
						xhat( b, i, j, z ) = ( in( b, i, j, z ) - mean( 0, i, j, z ) ) * inv_v;
					}
				}
			}
		}

		for ( int b = 0; b < in.size.b; b++ ){
			for (int i = 0; i < in.size.x; i++ ){
				for (int j = 0; j < in.size.y; j++ ){
					for (int z = 0; z < in.size.z; z++ ){
						out( b, i, j, z ) = gamma( 0, i, j, z ) * xhat( b, i, j, z ) + beta( 0, i, j, z );
					}
				}
			}
		}
	}

	void fix_weights()
	{
		for (int i=0; i<in.size.x *in.size.y *in.size.z; i++){
			gamma.data[i] -= lr * dgamma.data[i];
			beta.data[i] -= lr * dbeta.data[i];
		}
	}

	void calc_grads( TensorObject<float>& grad_next_layer )
	{
		for( int i=0; i < in.size.b *in.size.x *in.size.y *in.size.z; i++ ){
			dxhat.data[i] = grad_next_layer.data[i] * gamma.data[i];
		}

		for ( int i = 0; i < in.size.x; i++ ){
			for ( int j = 0; j < in.size.y; j++ ){
				for ( int z = 0; z < in.size.z; z++ ){

					float dbeta_sum = 0.0;
					float dgamma_sum = 0.0;
					float dx_sum = 0.0;
					float dxhat_sum = 0.0;
					
					for ( int b = 0; b < in.size.b; b++ ){

						float g = grad_next_layer( b, i, j, z );
						float xh = xhat( b, i, j, z );
						float dxh = dxhat( b, i, j, z );

						dbeta_sum += g;
						dgamma_sum += (xh * g);
						dx_sum += (xh * dxh);
						dxhat_sum += dxh;

					}

					for ( int b = 0; b < in.size.b; b++ ){
						grads_in( b, i, j, z ) = scale * inv_variance(0, i, j, z ) * ( (in.size.b * dxhat( b, i, j, z )) - dxhat_sum - (xhat( b, i, j, z ) * dx_sum));
					}
					dgamma( 0, i, j, z ) = dgamma_sum;
					dbeta( 0, i, j, z ) = dbeta_sum;
				}
			}
		}
	}
};
#pragma pack(pop)
