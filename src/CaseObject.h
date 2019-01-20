#pragma once
#include "TensorObject.h"

struct CaseObject
{
	TensorObject<float> data;
	TensorObject<float> out;
};

struct CasePaths
{
	std::string image_path;
	std::string label_path;
};
