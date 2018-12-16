#pragma once

enum class LayerType
{
	conv,
	dense,
	relu,
	pool,
	dropout_layer,
	softmax
};

enum class ActivationType
{
	relu,
	leaky_relu,
	sigmoid,
	softmax
};
