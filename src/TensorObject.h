#pragma once
#include "PointObject.h"
#include <cassert>
#include <vector>
#include <string.h>

template<typename T>
struct TensorObject
{
	T * data;

	tdsize size;

	TensorObject( int _x, int _y, int _z )
	{
		data = new T[_x * _y * _z];
		size.x = _x;
		size.y = _y;
		size.z = _z;
	}

	TensorObject( const TensorObject& other )
	{
		data = new T[other.size.x *other.size.y *other.size.z];
		memcpy(
			this->data,
			other.data,
			other.size.x *other.size.y *other.size.z * sizeof( T )
		);
		this->size = other.size;
	}

	TensorObject<T> operator+( TensorObject<T>& other )
	{
		TensorObject<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			clone.data[i] += other.data[i];
		return clone;
	}

	TensorObject<T> operator-( TensorObject<T>& other )
	{
		TensorObject<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			clone.data[i] -= other.data[i];
		return clone;
	}

	T& operator()( int _x, int _y, int _z )
	{
		return this->get( _x, _y, _z );
	}

	T& get( int _x, int _y, int _z )
	{
		assert( _x >= 0 && _y >= 0 && _z >= 0 );
		assert( _x < size.x && _y < size.y && _z < size.z );

		return data[
			_z * (size.x * size.y) +
				_y * (size.x) +
				_x
		];
	}

	void copy_from( std::vector<std::vector<std::vector<T>>> data )
	{
		int z = data.size();
		int y = data[0].size();
		int x = data[0][0].size();

		for ( int i = 0; i < x; i++ )
			for ( int j = 0; j < y; j++ )
				for ( int k = 0; k < z; k++ )
					get( i, j, k ) = data[k][j][i];
	}

	~TensorObject()
	{
		delete[] data;
	}
};
