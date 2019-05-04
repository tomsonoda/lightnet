#pragma once
#include <math.h>
#include <iostream>
#include <fstream>
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	void cudaMakeArray(float *gpu_array, int N);
	void batchNormalizationForwardGPU( float *in, float *out, float *mean, float *xmu, float *variance, float *inv_variance, float *xhat, float *gamma, float *beta, int batch_size, int in_size_x, int in_size_y, int in_size_z );
	void batchNormalizationUpdateWeightsGPU( float *gamma, float *beta, float *dxhat, float *dgamma, float *dbeta, float learning_rate, int in_size_z );
	void batchNormalizationBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *xmu, float *xhat, float *gamma, float *beta, float *dxhat, float *dx1, float *dgamma, float *dbeta, int batch_size, int in_size_x, int in_size_y, int in_size_z );
}
#endif

#pragma pack(push, 1)
struct LayerBatchNormalization
{
	LayerType type = LayerType::batch_normalization;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;

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

	float *gpu_gamma;
	float *gpu_dgamma;
	float *gpu_beta;
	float *gpu_dbeta;
	float *gpu_mean;
	float *gpu_xmu;
	float *gpu_variance;
	float *gpu_inv_variance;
	float *gpu_xhat;
	float *gpu_dxhat;
	float *gpu_dx1;

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
		for(int i=0; i<in_size.z; ++i){
			gamma.data[i] = 0.05 * rand() / float( RAND_MAX );
			beta.data[i]  = 0.05 * rand() / float( RAND_MAX );
		}
		memset( mean.data, 0, in.size.z );

#ifdef GPU_CUDA
		int data_size = in_size.b * in_size.x * in_size.y * in_size.z;
		gpu_cuda::cudaMakeArray(gpu_dz, data_size);
		gpu_cuda::cudaMakeArray(gpu_in, data_size);
		gpu_cuda::cudaMakeArray(gpu_out, data_size);
		gpu_cuda::cudaMakeArray(gpu_dz_in, data_size);

		gpu_cuda::cudaMakeArray(gpu_gamma, in_size.z);
		gpu_cuda::cudaMakeArray(gpu_dgamma, in_size.z);
		gpu_cuda::cudaMakeArray(gpu_beta, in_size.z);
		gpu_cuda::cudaMakeArray(gpu_dbeta, in_size.z);
		gpu_cuda::cudaMakeArray(gpu_mean, in_size.z);
		gpu_cuda::cudaMakeArray(gpu_xmu, data_size);
		gpu_cuda::cudaMakeArray(gpu_variance, in_size.z);
		gpu_cuda::cudaMakeArray(gpu_inv_variance, in_size.z);
		gpu_cuda::cudaMakeArray(gpu_xhat, data_size);
		gpu_cuda::cudaMakeArray(gpu_dxhat, data_size);
		gpu_cuda::cudaMakeArray(gpu_dx1, data_size);
#endif

	}

#ifdef GPU_CUDA

	void forwardGPU( float* in )
	{
		this->gpu_in = in;
		forwardGPU();
	}

	void forwardGPU()
	{
		gpu_cuda::batchNormalizationForwardGPU( gpu_in, gpu_out, gpu_mean, gpu_xmu, gpu_xhat, gpu_variance, gpu_inv_variance, gpu_dgamma, gpu_dbeta, in.size.b, in.size.x, in.size.y, in.size.z );
	}

	void updateWeightsGPU()
	{
		gpu_cuda::batchNormalizationUpdateWeightsGPU( gpu_gamma, gpu_beta, gpu_dxhat, gpu_dgamma, gpu_dbeta, lr, in.size.z );
	}

	void backwardGPU( float* dz_next_layer )
	{
		gpu_cuda::batchNormalizationBackwardGPU( dz_next_layer, gpu_dz_in, gpu_dz, gpu_xmu, gpu_xhat, gpu_gamma, gpu_beta, gpu_dxhat, gpu_dx1, gpu_dgamma, gpu_dbeta, in.size.b, in.size.x, in.size.y, in.size.z );
	}

#else

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		int filters = in.size.z;
		scale = 1.0f / (in.size.b * in.size.x * in.size.y);

		for ( int z = 0; z < filters; ++z ){
			float sum = 0;
			for ( int b = 0; b < in.size.b; ++b ){
				for ( int j = 0; j < in.size.y; ++j ){
					for ( int i = 0; i < in.size.x; ++i ){
							sum += in(b, i, j, z);
						}
					}
				}
				mean(0, 0, 0, z) = sum * scale;

				sum = 0;
				for ( int b = 0; b < in.size.b; ++b ){
					for ( int j = 0; j < in.size.y; ++j ){
						for ( int i = 0; i < in.size.x; ++i ){
							xmu( b, i, j, z ) = in(b, i, j, z) - mean(0, 0, 0, z);
							sum += pow( xmu( b, i, j, z ), 2 );
						}
					}
				}
				variance(0, 0, 0, z) = sum * scale;
				float invvar = 1.0f / sqrt(variance( 0, 0, 0, z )+0.00001f);
				float gmm = gamma( 0, 0, 0, z );
				float bt = beta( 0, 0, 0, z );
				inv_variance(0, 0, 0, z ) = invvar;

				for ( int b = 0; b < in.size.b; ++b ){
					for (int j = 0; j < in.size.y; ++j ){
						for (int i = 0; i < in.size.x; ++i ){
							float v = xmu( b, i, j, z ) * invvar;
							xhat( b, i, j, z ) = v;
							out( b, i, j, z ) = gmm * v + bt;
						}
					}
				}

			}
		}

		void updateWeights()
		{
			for( int i=0; i < in.size.z; ++i ){
				gamma.data[i] -= lr * dgamma.data[i];
				beta.data[i] -= lr * dbeta.data[i];
			}
		}

		void backward( TensorObject<float>& dz_next_layer )
		{

			for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
				dz_in.data[i] += dz_next_layer.data[i];
			}

			float bxy_inv = 1.0f / (float)(in.size.b * in.size.x * in.size.y);

			for ( int z = 0; z < in.size.z; ++z ){
				float dbeta_sum = 0.0;
				float dgamma_sum = 0.0;
				float dvariance_sum = 0.0;
				for ( int b = 0; b < in.size.b; ++b ){
					for ( int j = 0; j < in.size.y; ++j ){
						for ( int i = 0; i < in.size.x; ++i ){
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
				float gmm = gamma( 0, 0, 0, z );
				for ( int b = 0; b < in.size.b; ++b ){
					for ( int j = 0; j < in.size.y; ++j ){
						for ( int i = 0; i < in.size.x; ++i ){
							float v = dz_in( b, i, j, z ) * gmm;
							dxhat( b, i, j, z ) = v;
							divar += v * xmu( b, i, j, z );
						}
					}
				}

				float dmu = 0.0;
				float invvar = inv_variance( 0, 0, 0, z );
				float invvar_sqrt2 = -1. /(variance( 0, 0, 0, z )+0.00001f);

				for ( int b = 0; b < in.size.b; ++b ){
					for ( int j = 0; j < in.size.y; ++j ){
						for ( int i = 0; i < in.size.x; ++i ){
							// float dxmu1 = dxhat( b, i, j, z ) * invvar;
							float dxmu1 = dxhat( b, i, j, z );
							float dsqrtvar =  invvar_sqrt2 * divar;
							// float dvar = 0.5 * invvar * dsqrtvar;
							// float dxmu2 = 2 * xmu( b, i, j, z ) * bxy_inv * dvar;
							// float dxmu2 = xmu( b, i, j, z ) * bxy_inv * invvar * dsqrtvar;
							float dxmu2 = xmu( b, i, j, z ) * bxy_inv * dsqrtvar;
							float sum_dxmu = (dxmu1 + dxmu2) * invvar;
							dx1( b, i, j, z ) = sum_dxmu;
							dmu += -sum_dxmu;
						}
					}
				}

				float dx2 = dmu * bxy_inv;

				for ( int b = 0; b < in.size.b; ++b ){
					for ( int j = 0; j < in.size.y; ++j ){
						for ( int i = 0; i < in.size.x; ++i ){
							dz( b, i, j, z ) =  dx1( b, i, j, z ) + dx2;
						}
					}
				}

			}
		}

#endif

	void saveWeights( ofstream& fout )
	{
		int total_size = 0;
		int size = in.size.z * sizeof( float );
		fout.write(( char * )(gamma.data), size );
		total_size += size;
		fout.write(( char * )(beta.data), size );
		total_size += size;
		// cout << "- LayerBatchNormalization: " << to_string(total_size) << " bytes wrote." << endl;
	}

	void loadWeights( ifstream& fin )
	{
		int total_size = 0;
		int size = in.size.z * sizeof( float );
		fin.read(( char * )(gamma.data), size );
		total_size += size;
		fin.read(( char * )(beta.data), size );
		total_size += size;
		cout << "- LayerBatchNormalization: " << to_string(total_size) << " bytes read." << endl;
	}

};
#pragma pack(pop)
