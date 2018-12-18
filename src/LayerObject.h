#pragma once
#include "LayerType.h"
#include "TensorObject.h"

#pragma pack(push, 1)
struct LayerObject
{
	LayerType type;
	TensorObject<float> grads_in;
	TensorObject<float> in;
	TensorObject<float> out;
};
#pragma pack(pop)
