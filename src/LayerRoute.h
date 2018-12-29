#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerRoute
{
	LayerType type = LayerType::route;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	unsigned data_size;
	vector<LayerObject*> layers;
	vector<int> ref_layers;

	LayerRoute( vector<LayerObject*> layers, vector<int> ref_layers, TensorSize in_size )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		data_size = in_size.b * in_size.x * in_size.y * in_size.z * sizeof(float);
		this->layers = layers;
		this->ref_layers = ref_layers;
	}

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		int z_offset = 0;
		for( int i=0; i<ref_layers.size(); i++ ){
			TensorObject<float>& layer_in = layers[i]->out;
			for ( int b = 0; b < layer_in.size.b; b++ ){
				for ( int z = 0; z < layer_in.size.z; z++ ){
					for ( int y = 0; y < layer_in.size.y; y++ ){
						for ( int x = 0; x < layer_in.size.x; x++ ){
							out( b, x, y, z_offset+z ) = layer_in( b, x, y, z );
						}
					}
				}
				z_offset = layer_in.size.z;
			}
		}
	}

	void update_weights()
	{
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		int z_offset = 0;
		for( int i=0; i<ref_layers.size(); i++ ){
			TensorObject<float>& layer_dz = layers[i]->dz;
			for ( int b = 0; b < layer_dz.size.b; b++ ){
				for ( int z = 0; z < layer_dz.size.z; z++ ){
					for ( int y = 0; y < layer_dz.size.y; y++ ){
						for ( int x = 0; x < layer_dz.size.x; x++ ){
							layer_dz( b, x, y, z ) += dz_next_layer( b, x, y, z_offset+z );
						}
					}
				}
				z_offset = layer_dz.size.z;
			}
		}
	}
};
#pragma pack(pop)
