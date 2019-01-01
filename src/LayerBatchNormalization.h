#pragma once
#include "LayerObject.h"
#include <math.h>

#pragma pack(push, 1)
struct LayerBatchNormalization
{
	LayerType type = LayerType::batch_normalization;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;
	TensorObject<float> mean;
	TensorObject<float> xmu;
	TensorObject<float> variance;
	TensorObject<float> inv_variance;
	TensorObject<float> xhat;
	TensorObject<float> dxhat;
	TensorObject<float> dx1;
	TensorObject<float> gamma;
	TensorObject<float> dgamma;
	TensorObject<float> beta;
	TensorObject<float> dbeta;
	float scale;
	float lr;

	LayerBatchNormalization( TensorSize in_size, float learning_rate )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z ),
		mean( 1, 1, 1, in_size.z ),
		xmu( in_size.b, in_size.x, in_size.y, in_size.z ),
		variance( 1, 1, 1, in_size.z ),
		inv_variance( 1, 1, 1, in_size.z ),
		xhat( in_size.b, in_size.x, in_size.y, in_size.z ),
		dxhat( in_size.b, in_size.x, in_size.y, in_size.z ),
		dx1( in_size.b, in_size.x, in_size.y, in_size.z ),
		gamma( 1, 1, 1, in_size.z  ),
		dgamma( 1, 1, 1, in_size.z ),
		beta( 1, 1, 1, in_size.z ),
		dbeta( 1, 1, 1, in_size.z )
	{
		lr = learning_rate / (in.size.b * in.size.x * in.size.y);
		for(int i=0; i<in_size.z; i++){
			gamma.data[i] = 0.05 * rand() / float( RAND_MAX );
			beta.data[i]  = 0.05 * rand() / float( RAND_MAX );
		}
		memset( mean.data, 0, in.size.z );
	}

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		int filters = in.size.z;
		scale = 1.0f / (in.size.b * in.size.x * in.size.y);

		for ( int z = 0; z < filters; z++ ){
			float sum = 0;
			for ( int b = 0; b < in.size.b; b++ ){
				for ( int j = 0; j < in.size.y; j++ ){
					for ( int i = 0; i < in.size.x; i++ ){
							sum += in(b, i, j, z);
					}
				}
			}
			mean(0, 0, 0, z) = sum * scale;
		// }
		//
		// for ( int z = 0; z < filters; z++ ){
			sum = 0;
			for ( int b = 0; b < in.size.b; b++ ){
				for ( int j = 0; j < in.size.y; j++ ){
					for ( int i = 0; i < in.size.x; i++ ){
						xmu( b, i, j, z ) = in(b, i, j, z) - mean(0, 0, 0, z);
						sum += pow( xmu( b, i, j, z ), 2 );
					}
				}
			}
			variance(0, 0, 0, z) = sum * scale;
			inv_variance(0, 0, 0, z ) = 1.0f / sqrt(variance( 0, 0, 0, z )+0.00001f);

			for ( int b = 0; b < in.size.b; b++ ){
				for (int i = 0; i < in.size.x; i++ ){
					for (int j = 0; j < in.size.y; j++ ){
						float v = xmu( b, i, j, z ) * inv_variance( 0, 0, 0, z );
						xhat( b, i, j, z ) = v;
						out( b, i, j, z ) = gamma( 0, 0, 0, z ) * v + beta( 0, 0, 0, z );
					}
				}
			}
		}
	}

	void update_weights()
	{
		for (int i=0; i<in.size.z; i++){
			gamma.data[i] -= lr * dgamma.data[i];
			beta.data[i] -= lr * dbeta.data[i];
		}
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; i++ ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		float bxy_inv = 1.0f / (float)(in.size.b * in.size.x * in.size.y);

		for ( int z = 0; z < in.size.z; z++ ){
			float dbeta_sum = 0.0;
			float dgamma_sum = 0.0;
			float dvariance_sum = 0.0;
			for ( int b = 0; b < in.size.b; b++ ){
				for ( int i = 0; i < in.size.x; i++ ){
					for ( int j = 0; j < in.size.y; j++ ){
						float delta = dz_in( b, i, j, z );
						dbeta_sum += delta;
						dgamma_sum += xhat( b, i, j, z ) * delta;
						dvariance_sum += delta * xmu( b, i, j, z );
					}
				}
			}

			dbeta( 0, 0, 0, z ) = dbeta_sum;
			dgamma( 0, 0, 0, z ) = dgamma_sum;

			float divar = 0.0;
			for ( int b = 0; b < in.size.b; b++ ){
				for ( int i = 0; i < in.size.x; i++ ){
					for ( int j = 0; j < in.size.y; j++ ){
						dxhat( b, i, j, z ) = dz_in( b, i, j, z ) * gamma( 0, 0, 0, z );
						divar += dxhat( b, i, j, z ) * xmu( b, i, j, z );
					}
				}
			}

			float dmu = 0.0;
			for ( int b = 0; b < in.size.b; b++ ){
				for ( int i = 0; i < in.size.x; i++ ){
					for ( int j = 0; j < in.size.y; j++ ){
						float dxmu1 = dxhat( b, i, j, z ) * inv_variance( 0, 0, 0, z );
						float dsqrtvar = -1. /(variance( 0, 0, 0, z )+0.00001f) * divar;
						float dvar = 0.5 * inv_variance( 0, 0, 0, z ) * dsqrtvar;
						float dxmu2 = 2 * xmu( b, i, j, z ) * bxy_inv * dvar;
						dx1( b, i, j, z ) = (dxmu1 + dxmu2);
						dmu += -1 * (dxmu1 + dxmu2);
					}
				}
			}

			float dx2 = dmu * bxy_inv;

			for ( int b = 0; b < in.size.b; b++ ){
				for ( int i = 0; i < in.size.x; i++ ){
					for ( int j = 0; j < in.size.y; j++ ){
						dz( b, i, j, z ) =  dx1( b, i, j, z ) + dx2;
					}
				}
			}

		}
	}
};
#pragma pack(pop)
