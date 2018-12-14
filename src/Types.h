#pragma once

enum class LayerType
{
	conv,
	fc,
	relu,
	pool,
	dropout_layer,
	softmax,
	dense
};

enum class ActivationType
{
	relu,
	leaky_relu,
	sigmoid,
	softmax
};
