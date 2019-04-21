#pragma once
#include "TensorCoordinate.h"
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
		memset(data, 0x00, size.b * size.x * size.y * size.z * sizeof( T ));
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
		for ( int i = 0; i < other.size.b *other.size.x * other.size.y * other.size.z; ++i ){
			clone.data[i] += other.data[i];
		}
		return clone;
	}

	TensorObject<T> operator-( TensorObject<T>& other )
	{
		TensorObject<T> clone( *this );
		for ( int i = 0; i < other.size.b *other.size.x * other.size.y * other.size.z; ++i ){
			clone.data[i] -= other.data[i];
		}
		return clone;
	}

	void clear()
	{
		memset(data, 0x00, size.b * size.x * size.y * size.z * sizeof( T ));
	}

	T& operator()( int _b, int _x, int _y, int _z )
	{
		return this->get( _b, _x, _y, _z );
	}

	T& get( int _b, int _x, int _y, int _z )
	{
		// if( !(_b >= 0 && _x >= 0 && _y >= 0 && _z >= 0 && _b < size.b && _x < size.x && _y < size.y && _z < size.z) ){
		// 	printf("b=%d, x=%d, y=%d, z=%d, size_b=%d, size_x=%d, size_y=%d, size_z=%d\n", _b, _x, _y, _z, size.b, size.x, size.y, size.z);
		// }
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
