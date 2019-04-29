#pragma once
#include "LayerObject.h"

#pragma pack(push, 1)
struct LayerDetectObjects
{
	LayerType type = LayerType::detect_objects;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;
	unsigned _max_classes;
	unsigned _max_bounding_boxes;
	LayerDetectObjects( TensorSize in_size, uint16_t max_classes, uint16_t max_bounding_boxes )
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		_max_classes = max_classes;
		_max_bounding_boxes = max_bounding_boxes;
	}

#ifdef GPU_CUDA

	void forwardGPU( TensorObject<float>& in )
	{
		this->in = in;
		forwardGPU();
	}

	void forward()
	{
		for(int b = 0; b < in.size.b; ++b ){
			for( int i = 0; i < _max_bounding_boxes; i=i+(4+_max_classes)){
				out( b, i  , 0, 0 ) = 1.0f / (1.0f + exp( -in( b, i  , 0, 0 ) )); // x: sigmoid
				out( b, i+1, 0, 0 ) = 1.0f / (1.0f + exp( -in( b, i+1, 0, 0 ) )); // y: sigmoid
				out( b, i+2, 0, 0 ) = exp( in( b, i+2, 0, 0 ) ); // w: exp
				out( b, i+3, 0, 0 ) = exp( in( b, i+3, 0, 0 ) ); // h: exp
				for( int c = 0; c < _max_classes; ++c){
					out( b, i+4+c, 0, 0 ) = 1.0f / (1.0f + exp( -in( b, i+4+c , 0, 0 ) )); // id: sigmoid
				}
			}
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

		for(int b = 0; b < dz_in.size.b; ++b ){
			for( int i = 0; i < _max_bounding_boxes; i=i+(4+_max_classes)){
				dz( b, i  , 0, 0 ) = activator_derivative( out( b, i  , 0, 0 ) ) * dz_in( b, i  , 0, 0 ); // x: sigmoid derivative * grads
				dz( b, i+1, 0, 0 ) = activator_derivative( out( b, i+1 , 0, 0 ) ) * dz_in( b, i+1, 0, 0 ); // y: sigmoid derivative * grads
				dz( b, i+2, 0, 0 ) = exp( out( b, i+2, 0, 0 ) ) * dz_in( b, i+2, 0, 0 ); // w: exp * grads
				dz( b, i+3, 0, 0 ) = exp( out( b, i+3, 0, 0 ) ) * dz_in( b, i+3, 0, 0 ); // h: exp * grads
				for( int c = 0; c <_max_classes; ++c){
					dz( b, i+4+c, 0, 0 ) = activator_derivative( in( b, i+4+c , 0, 0 ) ) * dz_in( b, i+4+c , 0, 0 ); // id: sigmoid derivative * grads
				}
			}
		}
	}

#else

	void forward( TensorObject<float>& in )
	{
		this->in = in;
		forward();
	}

	void forward()
	{
		for(int b = 0; b < in.size.b; ++b ){
			for( int i = 0; i < _max_bounding_boxes; i=i+(4+_max_classes)){
				out( b, i  , 0, 0 ) = 1.0f / (1.0f + exp( -in( b, i  , 0, 0 ) )); // x: sigmoid
				out( b, i+1, 0, 0 ) = 1.0f / (1.0f + exp( -in( b, i+1, 0, 0 ) )); // y: sigmoid
				out( b, i+2, 0, 0 ) = exp( in( b, i+2, 0, 0 ) ); // w: exp
				out( b, i+3, 0, 0 ) = exp( in( b, i+3, 0, 0 ) ); // h: exp
				for( int c = 0; c < _max_classes; ++c){
					out( b, i+4+c, 0, 0 ) = 1.0f / (1.0f + exp( -in( b, i+4+c , 0, 0 ) )); // id: sigmoid
				}
			}
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

		for(int b = 0; b < dz_in.size.b; ++b ){
			for( int i = 0; i < _max_bounding_boxes; i=i+(4+_max_classes)){
				dz( b, i  , 0, 0 ) = activator_derivative( in( b, i  , 0, 0 ) ) * dz_in( b, i  , 0, 0 ); // x: sigmoid derivative * grads
				dz( b, i+1, 0, 0 ) = activator_derivative( in( b, i+1 , 0, 0 ) ) * dz_in( b, i+1, 0, 0 ); // y: sigmoid derivative * grads
				dz( b, i+2, 0, 0 ) = exp( in( b, i+2, 0, 0 ) ) * dz_in( b, i+2, 0, 0 ); // w: exp * grads
				dz( b, i+3, 0, 0 ) = exp( in( b, i+3, 0, 0 ) ) * dz_in( b, i+3, 0, 0 ); // h: exp * grads
				for( int c = 0; c < _max_classes; ++c){
					dz( b, i+4+c, 0, 0 ) = activator_derivative( in( b, i+4+c , 0, 0 ) ) * dz_in( b, i+4+c , 0, 0 ); // id: sigmoid derivative * grads
				}
			}
		}
	}

#endif

	float activator_function( float x )
	{
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig;
	}

	float activator_derivative( float x )
	{
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig * (1 - sig);
	}

};
#pragma pack(pop)
