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
	void cudaMakeArray(float *gpu_array, int N);
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

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;

	TensorObject<float> weights;
	TensorObject<float> dW;
	TensorObject<float> biases;
	TensorObject<float> dB;

	unsigned weigts_data_num;
	unsigned dw_data_size;
	unsigned dz_data_size;
	std::vector<GradientObject> gradients;
	float lr;
	float _decay;
	float _momentum;

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
	}

	int map( TensorCoordinate d )
	{
		return (d.b * (in.size.z * in.size.x * in.size.y)) + (d.z * (in.size.x * in.size.y)) + (d.y * (in.size.x)) + d.x;
	}

#ifdef GPU_CUDA

	void forwardGPU( float *in )
	{
		this->gpu_in = in;
		forwardGPU();
	}

	void forwardGPU()
	{
	}

	void updateWeightsGPU()
	{
	}

	void backwardGPU( float* dz_next_layer )
	{
	}

#else
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
		for (int i=0; i<weigts_data_num; ++i){
			weights.data[i] = weights.data[i] - lr * 	dW.data[i];
		}

		for (int i=0; i<out.size.x; ++i){
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
								float bias = biases( 0, 0, n, 0 );
								grad.grad = dzin;
								dW( 0, m, n, 0 ) += in( b, i, j, z ) * (grad.grad + grad.grad_prev * _momentum) + (_decay * w);
								dz( b, i, j, z ) += dzin * w;
								dB( 0, 0, n, 0 ) += dz( b, i, j, z ) * (grad.grad + grad.grad_prev * _momentum) + (_decay * bias);
							}
						}
					}
				}

				return 0;
			}));
		}

		for(auto && result: results){
			result.get();
		}
		results.erase(results.begin(), results.end());
	}

#endif

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
};
#pragma pack(pop)
