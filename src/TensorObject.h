#pragma once
#include "PointObject.h"
#include <cassert>
#include <vector>
#include <string.h>

template<typename T>
struct TensorObject
{
	T * data;
	TensorSize size;

	TensorObject( int _b, int _x, int _y, int _z )
	{
		data = new T[_b * _x * _y * _z];
		size.b = _b;
		size.x = _x;
		size.y = _y;
		size.z = _z;
	}

	TensorObject( const TensorObject& other )
	{
		data = new T[other.size.b *other.size.x *other.size.y *other.size.z];
		memcpy(
			this->data,
			other.data,
			other.size.b *other.size.x *other.size.y *other.size.z * sizeof( T )
		);
		this->size = other.size;
	}

	TensorObject<T> operator+( TensorObject<T>& other )
	{
		TensorObject<T> clone( *this );
		for ( int i = 0; i < other.size.b *other.size.x * other.size.y * other.size.z; i++ ){
			clone.data[i] += other.data[i];
		}
		return clone;
	}

	TensorObject<T> operator-( TensorObject<T>& other )
	{
		TensorObject<T> clone( *this );
		for ( int i = 0; i < other.size.b *other.size.x * other.size.y * other.size.z; i++ ){
			clone.data[i] -= other.data[i];
		}
		return clone;
	}

	T& operator()( int _b, int _x, int _y, int _z )
	{
		return this->get( _b, _x, _y, _z );
	}

	T& get( int _b, int _x, int _y, int _z )
	{
		assert( _b >= 0 && _x >= 0 && _y >= 0 && _z >= 0 );
		assert( _b < size.b && _x < size.x && _y < size.y && _z < size.z );

		return data[
			_b * (size.z * size.x * size.y) +
			_z * (size.x * size.y) +
			_y * (size.x) +
			_x
		];
	}

	~TensorObject()
	{
		delete[] data;
	}
};
