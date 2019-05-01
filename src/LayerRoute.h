#pragma once
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	void cudaMakeArray(float *gpu_array, int N);
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
	}

#ifdef GPU_CUDA

	void forwardGPU( float* in )
	{
		this->gpu_in = in;
		forwardGPU();
	}

	void forwardGPU()
	{
		/*
		int z_offset = 0;
		// printf("layers=%ld out_b_size=%d, out_x_size=%d, out_y_size=%d, out_z_size=%d, \n", ref_layers.size(), out.size.b, out.size.x, out.size.y, out.size.z);
		for( int i=0; i<ref_layers.size(); ++i ){
			TensorObject<float> layer_in = layers[ref_layers[i]]->out;
			// printf("          lin_b_size=%d, lin_x_size=%d, lin_y_size=%d, lin_z_size=%d, \n", layer_in.size.b, layer_in.size.x, layer_in.size.y, layer_in.size.z);
			for ( int b = 0; b < layer_in.size.b; ++b ){
				for ( int z = 0; z < layer_in.size.z; ++z ){
					for ( int y = 0; y < layer_in.size.y; y++ ){
						for ( int x = 0; x < layer_in.size.x; x++ ){
							out( b, x, y, z_offset+z ) = layer_in( b, x, y, z );
						}
					}
				}
			}
			z_offset = layer_in.size.z;
		}
		*/
	}

	void updateWeightsGPU()
	{
	}

	void backwardGPU( float* dz_next_layer )
	{
		/*
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		int z_offset = 0;
		// printf("layers=%ld dz_in=%d, dz_in=%d, dz_in=%d, dz_in=%d, \n", ref_layers.size(), dz_in.size.b, dz_in.size.x, dz_in.size.y, dz_in.size.z);
		for( int i=0; i<ref_layers.size(); ++i ){
			TensorObject<float>& layer_dz = layers[ref_layers[i]]->dz_in;
			// printf("  layer:%d: lin_b_size=%d, lin_x_size=%d, lin_y_size=%d, lin_z_size=%d, \n",ref_layers[i], layer_dz.size.b, layer_dz.size.x, layer_dz.size.y, layer_dz.size.z);
			for ( int b = 0; b < layer_dz.size.b; ++b ){
				for ( int z = 0; z < layer_dz.size.z; ++z ){
					for ( int y = 0; y < layer_dz.size.y; y++ ){
						for ( int x = 0; x < layer_dz.size.x; x++ ){
							layer_dz( b, x, y, z ) += dz_in( b, x, y, z_offset+z );
						}
					}
				}
			}
			z_offset = layer_dz.size.z;
		}
		*/
	}

#else

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		int z_offset = 0;
		for( int i=0; i<ref_layers.size(); ++i ){
			TensorObject<float> layer_in = layers[ref_layers[i]]->out;
			for ( int b = 0; b < layer_in.size.b; ++b ){
				for ( int z = 0; z < layer_in.size.z; ++z ){
					for ( int y = 0; y < layer_in.size.y; y++ ){
						for ( int x = 0; x < layer_in.size.x; x++ ){
							out( b, x, y, z_offset+z ) = layer_in( b, x, y, z );
						}
					}
				}
			}
			z_offset = layer_in.size.z;
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
		for( int i=0; i<ref_layers.size(); ++i ){
			TensorObject<float>& layer_dz = layers[ref_layers[i]]->dz_in;
			for ( int b = 0; b < layer_dz.size.b; ++b ){
				for ( int z = 0; z < layer_dz.size.z; ++z ){
					for ( int y = 0; y < layer_dz.size.y; y++ ){
						for ( int x = 0; x < layer_dz.size.x; x++ ){
							layer_dz( b, x, y, z ) += dz_in( b, x, y, z_offset+z );
						}
					}
				}
			}
			z_offset = layer_dz.size.z;
		}
	}

#endif

};
#pragma pack(pop)
