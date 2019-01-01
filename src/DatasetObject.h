#pragma once

struct DatasetObject
{
	uint32_t reverseUint32(uint32_t a)
	{
		return ((((a >> 24) & 0xff) << 0) |
			(((a >> 16) & 0xff) << 8) |
			(((a >> 8) & 0xff) << 16) |
			(((a >> 0) & 0xff) << 24));
	}

	uint8_t* readFileToBuffer( const char* szFile ){
		ifstream file( szFile, ios::binary | ios::ate );
		streamsize size = file.tellg();
		file.seekg( 0, ios::beg );

		if ( size == -1 ){
			return nullptr;
		}

		uint8_t* buffer = new uint8_t[size];
		file.read( (char*)buffer, size );
		return buffer;
	}

};
