#pragma once
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	float *cudaMakeArray( float *cpu_array, int N );
	void routeForwardGPU(float *in, float *out, int N, int in_size_x, int in_size_y, int in_size_z, int z_offset );
	void routeBackwardAddFirstArrayToSecondArrayGPU( float *dz_next_layer, float *dz_in, int N );
	void routeBackwardGPU( float *dz_in, float *dz, int N, int in_size_x, int in_size_y, int in_size_z, int z_offset );
}
#endif

#pragma pack(push, 1)
struct LayerRoute
{
	LayerType type = LayerType::route;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;

	int d_size;
	unsigned data_size;
	vector<LayerObject*> layers;
	vector<int> ref_layers;

	LayerRoute( vector<LayerObject*> layers, vector<int> ref_layers, TensorSize in_size )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		data_size = in_size.b * in_size.x * in_size.y * in_size.z * sizeof(float);
		this->layers = layers;
		this->ref_layers = ref_layers;

		#ifdef GPU_CUDA
				int d_size = in_size.b * in_size.x * in_size.y * in_size.z;
				gpu_dz = gpu_cuda::cudaMakeArray( dz.data, d_size );
				gpu_in = gpu_cuda::cudaMakeArray( in.data, d_size );
				gpu_out = gpu_cuda::cudaMakeArray( out.data, d_size );
				gpu_dz_in = gpu_cuda::cudaMakeArray( dz_in.data, d_size );
		#endif

	}

#ifdef GPU_CUDA


			void routePrintTensor( TensorObject<float>& data )
			{
				printf("route print\n");
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


	void forwardGPU( float *in, float *out, vector<float *>& outputArrays )
	{
		gpu_in = in;
		gpu_out = out;
		forwardGPU(outputArrays);
	}

	void forwardGPU( vector<float *>& outputArrays )
	{
		int z_offset = 0;
		for( unsigned i=0; i< ref_layers.size(); ++i ){
			float *layer_gpu_in = outputArrays[ref_layers[i]];
			TensorObject<float> layer_in = layers[ref_layers[i]]->out;

			int size = layer_in.size.b * layer_in.size.z * layer_in.size.y * layer_in.size.x;
			gpu_cuda::routeForwardGPU( layer_gpu_in, gpu_out, layer_in.size.x,  layer_in.size.y, layer_in.size.z, size, z_offset );
			z_offset += layer_in.size.z;
		}
	}

	void updateWeightsGPU()
	{
	}

	void backwardGPU( float* dz_next_layer, float *dz, float *dz_in, vector<float *>& dzInArrays )
	{
		this->gpu_dz = dz;
		this->gpu_dz_in = dz_in;
		backwardGPU( dz_next_layer, dzInArrays );
	}

	void backwardGPU( float* dz_next_layer, vector<float *>& dzInArrays )
	{
		int d_size_i = this->in.size.x * this->in.size.y * this->in.size.z;
		gpu_cuda::routeBackwardAddFirstArrayToSecondArrayGPU( dz_next_layer, gpu_dz_in, d_size_i );

		// gpu_cuda::cudaGetArray( this->dz_in.data, gpu_dz_in, this->dz_in.size.b * this->dz_in.size.x * this->dz_in.size.y * this->dz_in.size.z );
		// routePrintTensor( this->dz_in );

		int z_offset = 0;
		for( unsigned i=0; i<ref_layers.size(); ++i ){

			float* layer_gpu_dz_in = dzInArrays[ref_layers[i]];
			TensorObject<float>& layer_dz = layers[ref_layers[i]]->dz_in;

			int size = layer_dz.size.b * layer_dz.size.z * layer_dz.size.y * layer_dz.size.x;
			gpu_cuda::routeBackwardGPU( gpu_dz_in, layer_gpu_dz_in, size, layer_dz.size.x, layer_dz.size.y, layer_dz.size.z, z_offset );
			z_offset += layer_dz.size.z;
			// gpu_cuda::cudaGetArray( this->dz.data, layer_gpu_dz, this->in.size.b * this->in.size.x * this->in.size.y * this->in.size.z );
			// routePrintTensor( this->dz );

		}
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
		int z_offset = 0;
		for( unsigned i=0; i<ref_layers.size(); ++i ){
			TensorObject<float> layer_in = layers[ref_layers[i]]->out;
			for ( int b = 0; b < layer_in.size.b; ++b ){
				for ( int z = 0; z < layer_in.size.z; ++z ){
					for ( int y = 0; y < layer_in.size.y; ++y ){
						for ( int x = 0; x < layer_in.size.x; ++x ){
							out( b, x, y, z_offset+z ) = layer_in( b, x, y, z );
						}
					}
				}
			}
			z_offset += layer_in.size.z;
		}
	}

	void updateWeights()
	{
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		int z_offset = 0;
		for( unsigned i=0; i<ref_layers.size(); ++i ){
			TensorObject<float>& layer_dz = layers[ref_layers[i]]->dz_in;
			for ( int b = 0; b < layer_dz.size.b; ++b ){
				for ( int z = 0; z < layer_dz.size.z; ++z ){
					for ( int y = 0; y < layer_dz.size.y; ++y ){
						for ( int x = 0; x < layer_dz.size.x; ++x ){
							layer_dz( b, x, y, z ) += dz_in( b, x, y, z_offset+z );
						}
					}
				}
			}
			z_offset += layer_dz.size.z;
		}
	}

// #endif

};
#pragma pack(pop)
