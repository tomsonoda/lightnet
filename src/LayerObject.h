#pragma once
#include "LayerType.h"
#include "TensorObject.h"

#pragma pack(push, 1)
struct LayerObject
{
	LayerType type;

	TensorObject<float> dz;
	TensorObject<float> in;
	TensorObject<float> out;
	TensorObject<float> dz_in;

	float *gpu_dz;
	float *gpu_in;
	float *gpu_out;
	float *gpu_dz_in;
};
#pragma pack(pop)
