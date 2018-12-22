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
};
#pragma pack(pop)
