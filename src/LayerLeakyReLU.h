#pragma once
#include "LayerObject.h"

#ifdef GPU_CUDA
namespace gpu_cuda {
	void leakyReluForwardGPU(float *data_in, float *data_out, float *gpu_in, float *gpu_out, int N);
	void leakyReluBackwardGPU(float *data_in1, float *data_in2, float *data_in3, float *data_out,
		float *gpu_dz, float *gpu_in, float *gpu_out, float *gpu_dz_in,
		int N);
	void cudaMakeArray(float* gpu_out, int N);
} //namespace gpu
#endif

#pragma pack(push, 1)
struct LayerLeakyReLU
{
	LayerType type = LayerType::leaky_relu;
	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;
	unsigned data_size;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;
	float *gpu_dz_next_layer;

	LayerLeakyReLU( TensorSize in_size
	)
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		data_size = in_size.b * in_size.x * in_size.y * in_size.z;

#ifdef GPU_CUDA
		cudaMakeArray(gpu_dz, data_size);
		cudaMakeArray(gpu_in, data_size);
		cudaMakeArray(gpu_out, data_size);
		cudaMakeArray(gpu_dz_in, data_size);
		cudaMakeArray(gpu_dz_next_layer, data_size);
#endif

	}

		void forward(
			TensorObject<float>& in
		)
	{
		this->in = in;
		forward(
		);
	}

	void forward(
	)
	{
#ifdef GPU_CUDA
	gpu_cuda::leakyReluForwardGPU(in.data, out.data, gpu_in, gpu_out, data_size);
#else
		for( int i = 0; i < data_size; ++i ){
			float v = in.data[i];
			if ( v < 0 ){
				v = 0.1 * v;
			}
			out.data[i] = v;
		}
#endif
	}

	void update_weights()
	{
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
#ifdef GPU_CUDA
			gpu_cuda::leakyReluBackwardGPU(in.data, dz_next_layer.data, dz_in.data, dz.data,
				gpu_in, gpu_dz_next_layer, gpu_dz_in, gpu_dz,
				data_size);
#else
		// for( int i = 0; i < data_size ; ++i ){
		// 	dz_in.data[i] += dz_next_layer.data[i];
		// }
		// for( int i = 0; i < data_size; ++i ){
		// 	dz.data[i] +=  (in.data[i] < 0) ? (0.1) : (1.0 * dz_in.data[i]);
		// }
		for( int i = 0; i < data_size ; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
			dz.data[i] +=  (in.data[i] < 0) ? (0.1) : dz_in.data[i];
		}
#endif
	}
};
#pragma pack(pop)
