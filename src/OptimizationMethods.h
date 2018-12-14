#pragma once
#include "GradientObject.h"

#define LEARNING_RATE 0.001 // 0.01
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

static float update_weight( float w, GradientObject& grad, float multp = 1 )
{
	float m = (grad.grad + grad.oldgrad * MOMENTUM);
	// w -= LEARNING_RATE  * m * multp + LEARNING_RATE * WEIGHT_DECAY * w;
	w -= LEARNING_RATE  * ( (m * multp) + (WEIGHT_DECAY * w));
	return w;
}

static void update_gradient( GradientObject& grad )
{
	grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
}
