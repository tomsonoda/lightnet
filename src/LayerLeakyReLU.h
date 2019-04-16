#pragma once
#include "LayerObject.h"

#ifdef GPU_METAL
#include "mtlpp.hpp"
#endif

#ifdef GPU_METAL
mtlpp::Device device;
mtlpp::Library library;
mtlpp::Function metalFunc;
mtlpp::ComputePipelineState computePipelineState;
mtlpp::CommandQueue commandQueue;
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

	#ifdef GPU_METAL
	mtlpp::Buffer inBuffer;
	mtlpp::Buffer outBuffer;
	#endif

	LayerLeakyReLU( TensorSize in_size
	)
		:
		dz( in_size.b, in_size.x, in_size.y, in_size.z ),
		in( in_size.b, in_size.x, in_size.y, in_size.z ),
		out( in_size.b, in_size.x, in_size.y, in_size.z ),
		dz_in( in_size.b, in_size.x, in_size.y, in_size.z )
	{
		data_size = in_size.b * in_size.x * in_size.y * in_size.z;

		#ifdef GPU_METAL
			device = mtlpp::Device::CreateSystemDefaultDevice();
			assert(device);
			library = device.NewLibrary("gpu_metal.metallib", nullptr);
			assert(library);
			metalFunc = library.NewFunction("reakyRelu");
			assert(metalFunc);
			computePipelineState = device.NewComputePipelineState(metalFunc, nullptr);
			assert(computePipelineState);
			commandQueue = device.NewCommandQueue();
			assert(commandQueue);

			inBuffer = device.NewBuffer(sizeof(float) * data_size, mtlpp::ResourceOptions::StorageModeManaged);
			assert(inBuffer);
			outBuffer = device.NewBuffer(sizeof(float) * data_size, mtlpp::ResourceOptions::StorageModeManaged);
			assert(outBuffer);
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
		#ifndef GPU_METAL

		for( int i = 0; i < data_size; ++i ){
			float v = in.data[i];
			if ( v < 0 ){
				v = 0.1 * v;
			}
			out.data[i] = v;
		}

		#else
		// update input data
		{
				float* inData = static_cast<float*>(inBuffer.GetContents());
				for (uint32_t j=0; j<data_size; ++j){
					inData[j] = in.data[j];
				}
				inBuffer.DidModify(ns::Range(0, sizeof(float) * data_size));
		}

		mtlpp::CommandBuffer commandBuffer = commandQueue.CommandBuffer();
		assert(commandBuffer);

		mtlpp::ComputeCommandEncoder commandEncoder = commandBuffer.ComputeCommandEncoder();
		commandEncoder.SetBuffer(inBuffer, 0, 0);
		commandEncoder.SetBuffer(outBuffer, 0, 1);
		commandEncoder.SetComputePipelineState(computePipelineState);
		commandEncoder.DispatchThreadgroups(
				mtlpp::Size(1, 1, 1),
				mtlpp::Size(data_size, 1, 1));
		commandEncoder.EndEncoding();

		mtlpp::BlitCommandEncoder blitCommandEncoder = commandBuffer.BlitCommandEncoder();
		blitCommandEncoder.Synchronize(outBuffer);
		blitCommandEncoder.EndEncoding();

		commandBuffer.Commit();
		commandBuffer.WaitUntilCompleted();

		// read the data
		{
				float* outData = static_cast<float*>(outBuffer.GetContents());
				for (uint32_t j=0; j<data_size; ++j){
					out.data[j] = outData[j];
				}
		}

		#endif
	}

	void update_weights()
	{
	}

	void backward( TensorObject<float>& dz_next_layer )
	{
		for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
			dz_in.data[i] += dz_next_layer.data[i];
		}

		for( int i = 0; i < data_size; ++i ){
			dz.data[i] +=  (in.data[i] < 0) ? (0.1) : (1.0 * dz_in.data[i]);
		}
	}
};
#pragma pack(pop)
