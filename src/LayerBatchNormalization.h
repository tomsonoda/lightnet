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
	TensorObject<float> mean;
	TensorObject<float> dmean;
	TensorObject<float> variance;
	TensorObject<float> dvariance;
	TensorObject<float> inv_variance;
	TensorObject<float> xhat;
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
		mean( 1, 1, 1, in_size.z ),
		dmean( 1, 1, 1, in_size.z ),
		variance( 1, 1, 1, in_size.z ),
		dvariance( 1, 1, 1, in_size.z ),
		inv_variance( 1, 1, 1, in_size.z ),
		xhat( in_size.b, in_size.x, in_size.y, in_size.z ),
		gamma( 1, 1, 1, in_size.z  ),
		dgamma( 1, 1, 1, in_size.z ),
		beta( 1, 1, 1, in_size.z ),
		dbeta( 1, 1, 1, in_size.z )
	{
		lr = learning_rate / (float)in_size.b;
		for(int i=0; i<in_size.z; i++){
			gamma.data[i] = 0.05 * rand() / float( RAND_MAX );
			beta.data[i]  = 0.05 * rand() / float( RAND_MAX );
		}
	}

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		int filters = in.size.z;
		int batch = in.size.b;

		scale = 1.0f / (batch * in.size.x * in.size.y);
		memset( mean.data, 0, in.size.z );
		for ( int z = 0; z < filters; z++ ){
			float sum = 0;
			for ( int i = 0; i < in.size.x; i++ ){
				for ( int j = 0; j < in.size.y; j++ ){
					for ( int b = 0; b < in.size.b; b++ ){
							sum += in(b, i, j, z);
					}
				}
			}
			mean(0, 0, 0, z) = sum * scale;
		}

		scale = 1./(batch * in.size.x * in.size.y - 1);
		memset( mean.data, 0, in.size.z );
		for ( int z = 0; z < filters; z++ ){
			float sum = 0;
			for ( int i = 0; i < in.size.x; i++ ){
				for ( int j = 0; j < in.size.y; j++ ){
					for ( int b = 0; b < in.size.b; b++ ){
						sum += pow( (in(b, i, j, z) - mean(0, 0, 0, z)), 2 );
					}
				}
			}
			variance(0, 0, 0, z) = sum * scale;
			inv_variance(0, 0, 0, z ) = 1.0f / sqrt(variance( 0, 0, 0, z )+0.00001f);

			for ( int b = 0; b < in.size.b; b++ ){
				for (int i = 0; i < in.size.x; i++ ){
					for (int j = 0; j < in.size.y; j++ ){
						float v = ( in( b, i, j, z ) - mean( 0, 0, 0, z ) ) * inv_variance( 0, 0, 0, z );
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
		for ( int z = 0; z < in.size.z; z++ ){
			float dbeta_sum = 0.0;
			float dgamma_sum = 0.0;
			float dvariance_sum = 0.0;
			for ( int b = 0; b < in.size.b; b++ ){
				for ( int i = 0; i < in.size.x; i++ ){
					for ( int j = 0; j < in.size.y; j++ ){
						float delta = dz_next_layer( b, i, j, z );
						float xh = xhat( b, i, j, z ); // norm
						dbeta_sum += delta;
						dgamma_sum += (xh * delta);
						dvariance_sum += delta*(in( b, i, j, z ) - mean(0, 0, 0, z));
					}
				}
			}
			dbeta( 0, 0, 0, z ) = dbeta_sum;
			dgamma( 0, 0, 0, z ) = dgamma_sum;
			dmean( 0, 0, 0, z ) = dbeta_sum * inv_variance( 0, 0, 0, z );
			dvariance( 0, 0, 0, z ) = dvariance_sum * (-0.5 * pow(variance( 0, 0, 0, z ) + 0.00001f, (float)(-3./2.)));

			for ( int b = 0; b < in.size.b; b++ ){
				for ( int i = 0; i < in.size.x; i++ ){
					for ( int j = 0; j < in.size.y; j++ ){
						float bxy = (float)(in.size.b * in.size.x * in.size.y);
						dz( b, i, j, z ) = dz_next_layer( b, i, j, z ) * inv_variance( 0, 0, 0, z ) + dvariance( 0, 0, 0, z ) * 2.0 * (in ( b, i, j, z) - mean(0, 0, 0, z)) / bxy + dmean( 0, 0, 0, z )/bxy;
					}
				}
			}
		}
	}
};
#pragma pack(pop)
