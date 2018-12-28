#pragma once

struct GradientObject
{
	float grad;
	float grad_prev;
	GradientObject()
	{
		grad = 0;
		grad_prev = 0;
	}
};
