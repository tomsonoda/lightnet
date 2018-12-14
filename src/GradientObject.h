#pragma once


struct GradientObject
{
	float grad;
	float oldgrad;
	GradientObject()
	{
		grad = 0;
		oldgrad = 0;
	}
};
