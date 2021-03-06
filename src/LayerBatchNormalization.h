#pragma once
#include <math.h>
#include <iostream>
#include <fstream>
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	void cudaGetArray( float *cpu_array, float *gpu_array, size_t N );
	float *cudaMakeArray( float *cpu_array, int N );
	void cudaClearArray( float *gpu_array, int N );
	void batchNormalizationForwardGPU( float *in, float *out, float *mean, float *xmu, float *variance, float *inv_variance, float *xhat, float *gamma, float *beta, int batch_size, int in_size_x, int in_size_y, int in_size_z );
	void batchNormalizationUpdateWeightsGPU( float *gamma, float *beta, float *dgamma, float *dbeta, float learning_rate, int in_size_z );
	void batchNormalizationBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *xmu, float *variance, float *inv_variance, float *xhat, float *gamma, float *beta, float *dxhat, float *dx1, float *dgamma, float *dbeta, int batch_size, int in_size_x, int in_size_y, int in_size_z );
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
		gpu_dz = gpu_cuda::cudaMakeArray( dz.data, data_size );
		gpu_in = gpu_cuda::cudaMakeArray( in.data, data_size);
		gpu_out = gpu_cuda::cudaMakeArray( out.data, data_size );
		gpu_dz_in = gpu_cuda::cudaMakeArray( dz_in.data, data_size );

		gpu_gamma = gpu_cuda::cudaMakeArray( gamma.data, in_size.z );
		gpu_dgamma = gpu_cuda::cudaMakeArray( NULL, in_size.z );
		gpu_beta = gpu_cuda::cudaMakeArray( beta.data, in_size.z );
		gpu_dbeta = gpu_cuda::cudaMakeArray( NULL, in_size.z );

		gpu_mean = gpu_cuda::cudaMakeArray( NULL, in_size.z );
		gpu_xmu = gpu_cuda::cudaMakeArray( NULL, data_size );
		gpu_variance = gpu_cuda::cudaMakeArray( NULL, in_size.z );
		gpu_inv_variance = gpu_cuda::cudaMakeArray( NULL, in_size.z );
		gpu_xhat = gpu_cuda::cudaMakeArray( NULL, data_size );
		gpu_dxhat = gpu_cuda::cudaMakeArray( NULL, data_size );
		gpu_dx1 = gpu_cuda::cudaMakeArray( NULL, data_size );
#endif

	}

#ifdef DEBUG

	void batchNormalizationPrintTensor( TensorObject<float>& data )
	{
		int mx = data.size.x;
		int my = data.size.y;
		int mz = data.size.z;
		int mb = data.size.b;

		for ( int b = 0; b < mb; ++b ){
			printf( "[Batch %d]\n", b );
			for ( int z = 0; z < mz; ++z ){
				for ( int y = 0; y < my; y++ ){
					for ( int x = 0; x < mx; x++ ){
						printf( "%.3f \t", (float)data( b, x, y, z ) );
					}
					printf( "\n" );
				}
				printf( "\n" );
			}
		}
	}

#endif


#ifdef GPU_CUDA

	void forwardGPU( float *in, float *out )
	{
		gpu_in = in;
		gpu_out = out;

		// gpu_cuda::cudaGetArray( this->in.data, in, this->in.size.b * this->in.size.x * this->in.size.y * this->in.size.z );
		// forward();
		forwardGPU();
	}

	void forwardGPU()
	{
		gpu_cuda::batchNormalizationForwardGPU( gpu_in, gpu_out, gpu_mean, gpu_xmu, gpu_variance, gpu_inv_variance, gpu_xhat, gpu_gamma, gpu_beta, in.size.b, in.size.x, in.size.y, in.size.z );
	}

	void updateWeightsGPU()
	{
		// updateWeights();

		gpu_cuda::batchNormalizationUpdateWeightsGPU( gpu_gamma, gpu_beta, gpu_dgamma, gpu_dbeta, lr, in.size.z );
	}

	void backwardGPU( float* dz_next_layer, float *dz, float *dz_in )
	{
		// gpu_cuda::cudaGetArray( this->dz_in.data, dz_next_layer, this->dz_in.size.b * this->dz_in.size.x * this->dz_in.size.y * this->dz_in.size.z );
		// backward();

		this->gpu_dz = dz;
		this->gpu_dz_in = dz_in;
		backwardGPU( dz_next_layer );

	}

	void backwardGPU( float* dz_next_layer )
	{
		gpu_cuda::batchNormalizationBackwardGPU( dz_next_layer, gpu_dz_in, gpu_dz, gpu_xmu, gpu_variance, gpu_inv_variance, gpu_xhat, gpu_gamma, gpu_beta, gpu_dxhat, gpu_dx1, gpu_dgamma, gpu_dbeta, in.size.b, in.size.x, in.size.y, in.size.z );
	}

	TensorObject<float> getOutFromGPU(){
		gpu_cuda::cudaGetArray( out.data, gpu_out, out.size.b*out.size.x*out.size.y*out.size.z );
		return out;
	}
	void clearArrayGPU(float *dz_)
	{
		this->gpu_dz = dz_;
		gpu_cuda::cudaClearArray( gpu_dz_in, dz_in.size.b*dz_in.size.x*dz_in.size.y*dz_in.size.z );
		gpu_cuda::cudaClearArray( gpu_dz, dz.size.b*dz.size.x*dz.size.y*dz.size.z );
	}

#endif

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

	void backward()
	{
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

// #endif

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
