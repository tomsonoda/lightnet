#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "LayerObject.h"
#include "GradientObject.h"
#include "ThreadPool.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	void cudaGetArray( float *cpu_array, float *gpu_array, size_t N );
	float *cudaMakeArray( float *cpu_array, int N );
	void cudaClearArray( float *gpu_array, int N );
	void denseForwardGPU( float *in, float *out, float *weights, float *biases, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z );
	void denseUpdateWeightsGPU( float *weights, float *biases, float *gradients, float *dW, float *dB, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, float learning_rate, float momentum );
	void denseBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, float *weights, float *biases, float *gradients, float *dW, float *dB, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, float momentum, float decay );
}
#endif

#pragma pack(push, 1)
struct LayerDense
{
	LayerType type = LayerType::dense;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	TensorObject<float> weights;
	TensorObject<float> dW;
	TensorObject<float> biases;
	TensorObject<float> dB;
	vector<GradientObject> gradients;

	unsigned weigts_data_num;
	unsigned dw_data_size;
	unsigned dz_data_size;

	float lr;
	float _decay;
	float _momentum;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;

	float *gpu_weights;
	float *gpu_dW;
	float *gpu_biases;
	float *gpu_dB;
	float *gpu_gradients;

	LayerDense( TensorSize in_size, int out_size, float learning_rate, float decay, float momentum)
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, out_size, 1, 1 ),
		dz_in( in_size.b, out_size, 1, 1 ),
		weights( 1, in_size.x*in_size.y*in_size.z, out_size, 1 ),
		dW( 1, in_size.x*in_size.y*in_size.z, out_size, 1 ),
		biases( 1, 1, out_size, 1 ),
		dB( 1, 1, out_size, 1 )
	{
		gradients = std::vector<GradientObject>( out_size * in_size.b );
		lr = learning_rate / (float)in_size.b;
		int maxval = in_size.x * in_size.y * in_size.z;
		_decay = decay;
		_momentum = momentum;

		weigts_data_num = in_size.x * in_size.y * in_size.z * out_size;
		dw_data_size = in_size.x * in_size.y * in_size.z * out_size * sizeof( float );
		dz_data_size = in_size.b * in_size.x * in_size.y * in_size.z * sizeof( float );

		for(int i=0; i<out_size; ++i){
			for(int h=0; h<in_size.x*in_size.y*in_size.z; ++h){
				// weights( 0, h, i, 0 ) = 0.05 * rand() / float( RAND_MAX );
				weights( 0, h, i, 0 ) = (2.19722f / maxval) * rand() / float( RAND_MAX );
			}
		}

		for(int i=0; i<out_size; ++i){
			biases( 0, 0, i, 0 ) = 0.0;
		}

		for(int i=0; i<out_size * in_size.b; ++i){
			gradients[i].grad = 0;
			gradients[i].grad_prev = 0;
		}

#ifdef GPU_CUDA
		// int d_size = in_size.b * in_size.x * in_size.y * in_size.z;
		// gpu_dz = gpu_cuda::cudaMakeArray( NULL, d_size );
		// gpu_in = gpu_cuda::cudaMakeArray( NULL, d_size );
		int o_size = in_size.b * out_size;
		// gpu_out   = gpu_cuda::cudaMakeArray( NULL, o_size );
		gpu_dz_in = gpu_cuda::cudaMakeArray( NULL, o_size );
		gpu_weights = gpu_cuda::cudaMakeArray( weights.data, weigts_data_num );
		gpu_dW = gpu_cuda::cudaMakeArray( NULL, weigts_data_num );
		gpu_biases = gpu_cuda::cudaMakeArray( NULL, out_size );
		gpu_dB = gpu_cuda::cudaMakeArray( NULL, out_size );
		gpu_gradients = gpu_cuda::cudaMakeArray( NULL, o_size * 2 );  // 2n:current, 2n+1:prev
#endif

	}

	int map( TensorCoordinate d )
	{
		return (d.b * (in.size.z * in.size.x * in.size.y)) + (d.z * (in.size.x * in.size.y)) + (d.y * (in.size.x)) + d.x;
	}

#ifdef GPU_CUDA

	void forwardGPU( float *in, float *out )
	{
		// gpu_cuda::cudaGetArray( this->in.data, in, this->in.size.b * this->in.size.x * this->in.size.y * this->in.size.z );
		// forward();
		this->gpu_in = in;
		this->gpu_out = out;
		forwardGPU();
	}

	void forwardGPU()
	{
		gpu_cuda::denseForwardGPU( gpu_in, gpu_out, gpu_weights, gpu_biases, in.size.b, in.size.x, in.size.y, in.size.z, out.size.x, out.size.y, out.size.z );
	}

	void updateWeightsGPU()
	{
		// updateWeights();
		gpu_cuda::denseUpdateWeightsGPU( gpu_weights, gpu_biases, gpu_gradients, gpu_dW, gpu_dB, in.size.b, in.size.x, in.size.y, in.size.z, out.size.x, out.size.y, out.size.z, lr, _momentum );
	}

	void backwardGPU( float* dz_next_layer, float *dz )
	{
		// gpu_cuda::cudaGetArray( this->dz_in.data, dz_next_layer, this->dz_in.size.b * this->dz_in.size.x * this->dz_in.size.y * this->dz_in.size.z );
		// backward();
		this->gpu_dz = dz;
		backwardGPU( dz_next_layer );
	}

	void backwardGPU( float* dz_next_layer )
	{
		gpu_cuda::cudaClearArray( gpu_dW, in.size.x * in.size.y * in.size.z * out.size.x );
		gpu_cuda::cudaClearArray( gpu_dB, out.size.x );
		gpu_cuda::denseBackwardGPU( dz_next_layer, gpu_dz_in, gpu_dz, gpu_in, gpu_weights, gpu_biases, gpu_gradients, gpu_dW, gpu_dB, in.size.b, in.size.x, in.size.y, in.size.z, out.size.x, out.size.y, out.size.z, _momentum, _decay );
	}

	void clearArrayGPU(float *dz_)
	{
		this->gpu_dz = dz_;
		gpu_cuda::cudaClearArray( gpu_dz_in, dz_in.size.b*dz_in.size.x*dz_in.size.y*dz_in.size.z );
		gpu_cuda::cudaClearArray( gpu_dz, dz.size.b*dz.size.x*dz.size.y*dz.size.z );
	}

	TensorObject<float> getOutFromGPU()
	{
		gpu_cuda::cudaGetArray( out.data, gpu_out, out.size.b*out.size.x*out.size.y*out.size.z );
		return out;
	}

#endif
// CPU
	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for ( int b = 0; b < in.size.b; ++b ){
			for ( int n = 0; n < out.size.x; ++n ){
				float sum = 0;
				for ( int z = 0; z < in.size.z; ++z ){
					for ( int j = 0; j < in.size.y; ++j ){
						for ( int i = 0; i < in.size.x; ++i ){
							int m = map( { 0, i, j, z } );
							sum += in( b, i, j, z ) * weights( 0, m, n, 0 );
						}
					}
				}
				out( b, n, 0, 0 ) = sum + biases( 0, 0, n, 0);
			}
		}
	}

	void updateWeights()
	{
		for ( unsigned i=0; i<weigts_data_num; ++i){
			weights.data[i] = weights.data[i] - lr * 	dW.data[i];
		}

		for ( int i=0; i<out.size.x; ++i){
			biases.data[i] = biases.data[i] - lr * 	dB.data[i];
		}

		for ( int i = 0; i < out.size.x * in.size.b; ++i ){
				GradientObject& grad = gradients[ i ];
				grad.grad_prev = (grad.grad + grad.grad_prev * _momentum);
		}
	}

	void backward( TensorObject<float>& dz_next_layer, ThreadPool& thread_pool )
	{
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		std::vector< std::future<int> > results;

		memset( dW.data, 0, dw_data_size );
		memset( dB.data, 0, out.size.x * sizeof(float) );
		for ( int n = 0; n < out.size.x; ++n ){
			results.emplace_back( thread_pool.enqueue([&, n] {

				for ( int z = 0; z < in.size.z; ++z ){
					for ( int j = 0; j < in.size.y; ++j ){
						for ( int i = 0; i < in.size.x; ++i ){
							int m = map( { 0, i, j, z } );

							for( int b = 0; b < in.size.b; ++b ){
								GradientObject& grad = gradients[ n*in.size.b + b ];
								float dzin = dz_in( b, n, 0, 0 );
								float w = weights(0, m, n, 0);
								grad.grad = dzin;
								dz( b, i, j, z ) += dzin * w;
								dW( 0, m, n, 0 ) += in( b, i, j, z ) * (grad.grad + grad.grad_prev * _momentum) + (_decay * w);
							}
						}
					}
				}

				for( int b = 0; b < in.size.b; ++b ){
					dB( 0, 0, n, 0 ) += dz_in( b, n, 0, 0 );
				}

				return 0;

			}));
		}

		for(auto && result: results){
			result.get();
		}
		results.erase(results.begin(), results.end());
	}

	void backward()
	{
		memset( dW.data, 0, dw_data_size );
		memset( dB.data, 0, out.size.x * sizeof(float) );

		// for ( int n = 0; n < out.size.x; ++n ){
		// 	for( int b = 0; b < in.size.b; ++b ){
		// 		GradientObject& grad = gradients[ n*in.size.b + b ];
		// 		// printf("CPU: grad=%lf, prev_grad=%lf\n", grad.grad, grad.grad_prev);
		// 	}
		// }

		for ( int n = 0; n < out.size.x; ++n ){

				for ( int z = 0; z < in.size.z; ++z ){
					for ( int j = 0; j < in.size.y; ++j ){
						for ( int i = 0; i < in.size.x; ++i ){
							int m = map( { 0, i, j, z } );

							for( int b = 0; b < in.size.b; ++b ){
								GradientObject& grad = gradients[ n*in.size.b + b ];
								float dzin = dz_in( b, n, 0, 0 );
								float w = weights(0, m, n, 0);
								grad.grad = dzin;
								dz( b, i, j, z ) += dzin * w;
								dW( 0, m, n, 0 ) += in( b, i, j, z ) * (grad.grad + grad.grad_prev * _momentum) + (_decay * w);
							}
						}
					}
				}

				for( int b = 0; b < in.size.b; ++b ){
					dB( 0, 0, n, 0 ) += dz_in( b, n, 0, 0 );
					// printf("dB=%lf dz_in=%lf\n", dB( 0, 0, n, 0 ), dz_in( b, n, 0, 0 ));
				}
		}
	}

// #endif

	void saveWeights( ofstream& fout )
	{
		int total_size = 0;
		for ( int i = 0; i < out.size.x * in.size.b ; ++i ){
			GradientObject grad = gradients[i];
			fout.write(( char * ) &(grad.grad_prev), sizeof( float ) );
			total_size += sizeof(float);
		}

		int size = weigts_data_num * sizeof( float );
		fout.write(( char * )(weights.data), size );
		total_size += size;
		// cout << "- LayerDense             : " << to_string(total_size) << " bytes wrote." << endl;
	}

	void loadWeights( ifstream& fin )
	{
		int total_size = 0;
		for ( int i = 0; i < out.size.x * in.size.b ; ++i ){
			GradientObject grad = gradients[i];
			fin.read(( char * ) &(grad.grad_prev), sizeof( float ) );
			total_size += sizeof(float);
		}

		int size = weigts_data_num * sizeof( float );
		fin.read(( char * )(weights.data), size );
		total_size += size;
		cout << "- LayerDense             : " << to_string(total_size) << " bytes read." << endl;
	}

#if DEBUG

	TensorObject<float> getWeights()
	{
		return weights;
	}

	TensorObject<float> getWeightsFromGPU()
	{
		gpu_cuda::cudaGetArray( weights.data, gpu_weights, weights.size.b * weights.size.x * weights.size.y * weights.size.z );
		return weights;
	}

	TensorObject<float> getBiases()
	{
		return biases;
	}

	TensorObject<float> getBiasesFromGPU()
	{
		gpu_cuda::cudaGetArray( biases.data, gpu_biases, biases.size.b * biases.size.x * biases.size.y * biases.size.z );
		return biases;
	}

	TensorObject<float> getDW()
	{
		return dW;
	}

	TensorObject<float> getDWFromGPU()
	{
		gpu_cuda::cudaGetArray( dW.data, gpu_dW, dW.size.b * dW.size.x * dW.size.y * dW.size.z );
		return dW;
	}

	TensorObject<float> getDB()
	{
		return dB;
	}

	TensorObject<float> getDBFromGPU()
	{
		gpu_cuda::cudaGetArray( dB.data, gpu_dB, dB.size.b * dB.size.x * dB.size.y * dB.size.z );
		return dB;
	}

	TensorObject<float> getDzInFromGPU()
	{
		gpu_cuda::cudaGetArray( dz_in.data, gpu_dz_in, dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z );
		return dz_in;
	}

	TensorObject<float> getDzFromGPU(){
		gpu_cuda::cudaGetArray( dz.data, gpu_dz, dz.size.b*dz.size.x*dz.size.y*dz.size.z );
		return dz;
	}

#endif

};
#pragma pack(pop)
