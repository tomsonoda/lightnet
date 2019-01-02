#pragma once
#include "CaseObject.h"
#include "JSONObject.h"
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

	vector<CaseObject> readCasesMNIST( JSONObject *data_json, vector <json_token_t*> data_tokens, string mode )
	{
		DatasetObject *dataset = new DatasetObject();
		vector<CaseObject> cases;

		uint8_t* images;
		uint8_t* labels;

		if(mode=="train"){
			string train_dir = data_json->getChildValueForToken(data_tokens[0], "train_dir");   // data_tokens[0] := json root
			images = dataset->readFileToBuffer( (train_dir + "train-images-idx3-ubyte").c_str() );
			labels = dataset->readFileToBuffer( (train_dir + "train-labels-idx1-ubyte").c_str() );
		}else{
			string test_dir = data_json->getChildValueForToken(data_tokens[0], "test_dir");   // data_tokens[0] := json root
			images = dataset->readFileToBuffer( (test_dir + "t10k-images-idx3-ubyte").c_str() );
			labels = dataset->readFileToBuffer( (test_dir + "t10k-labels-idx1-ubyte").c_str() );
		}

		uint32_t case_count = dataset->reverseUint32( *(uint32_t*)(images + 4) );

		for (int i=0; i<case_count; ++i){
			CaseObject c {TensorObject<float>( 1, 28, 28, 1 ), TensorObject<float>( 1, 10, 1, 1 )};
			uint8_t* img = images + 16 + i * (28 * 28);
			uint8_t* label = labels + 8 + i;
			for ( int x = 0; x < 28; x++ ){
				for ( int y = 0; y < 28; y++ ){
					c.data( 0, x, y, 0 ) = img[x + y * 28] / 255.f;
				}
			}
			for ( int b = 0; b < 10; ++b ){
				c.out( 0, b, 0, 0 ) = *label == b ? 1.0f : 0.0f;
			}
			cases.push_back( c );
		}

		delete[] images;
		delete[] labels;
		return cases;
	}

	vector<CaseObject> readCasesCifar10( JSONObject *data_json, vector<json_token_t*> data_tokens, string mode )
	{
		DatasetObject *dataset = new DatasetObject();
		vector<CaseObject> cases;

		uint8_t* buffer;
		vector<uint8_t> v;
		streamsize file_size;
		if(mode=="train"){
			string train_dir = data_json->getChildValueForToken(data_tokens[0], "train_dir");   // data_tokens[0] := json root

			const char *file_name = (train_dir + "data_batch_1.bin").c_str();
			ifstream file1( file_name, ios::binary | ios::ate );
			long file_size1 = file1.tellg();
			uint8_t* buffer1 = dataset->readFileToBuffer( file_name );

			file_name = (train_dir + "data_batch_2.bin").c_str();
			ifstream file2( file_name, ios::binary | ios::ate );
			long file_size2 = file2.tellg();
			uint8_t* buffer2 = dataset->readFileToBuffer( file_name );

			file_name = (train_dir + "data_batch_3.bin").c_str();
			ifstream file3( file_name, ios::binary | ios::ate );
			long file_size3 = file3.tellg();
			uint8_t* buffer3 = dataset->readFileToBuffer( file_name );

			file_name = (train_dir + "data_batch_4.bin").c_str();
			ifstream file4( file_name, ios::binary | ios::ate );
			long file_size4 = file4.tellg();
			uint8_t* buffer4 = dataset->readFileToBuffer( file_name );

			file_name = (train_dir + "data_batch_5.bin").c_str();
			ifstream file5( file_name, ios::binary | ios::ate );
			long file_size5 = file5.tellg();
			uint8_t* buffer5 = dataset->readFileToBuffer( file_name );

			int pointer = 0;
			file_size = file_size1 + file_size2 + file_size3 + file_size4 + file_size5;
			buffer = new uint8_t[file_size];
			memcpy(buffer, buffer1, file_size1);
			pointer += file_size1;
			memcpy(buffer+pointer, buffer2, file_size2);
			pointer += file_size2;
			memcpy(buffer+pointer, buffer3, file_size3);
			pointer += file_size3;
			memcpy(buffer+pointer, buffer4, file_size4);
			pointer += file_size4;
			memcpy(buffer+pointer, buffer5, file_size5);

			delete[] buffer1;
			delete[] buffer2;
			delete[] buffer3;
			delete[] buffer4;
			delete[] buffer5;

		}else{
			string test_dir = data_json->getChildValueForToken(data_tokens[0], "test_dir");   // data_tokens[0] := json root
			const char *file_name = (test_dir + "test_batch.bin").c_str();
			ifstream file( file_name, ios::binary | ios::ate );
			file_size = file.tellg();
			buffer = dataset->readFileToBuffer( file_name );
		}
		uint32_t case_count = file_size / (32 * 32 * 3 + 1);
		for (int i=0; i<case_count; ++i){
			CaseObject c {TensorObject<float>( 1, 32, 32, 3 ), TensorObject<float>( 1, 10, 1, 1 )};
			uint8_t* img = buffer + i * (32 * 32 * 3 + 1) + 1;
			uint8_t* label = buffer + i * (32 * 32 * 3 + 1);
			for(int z = 0; z < 3; ++z ){
				for ( int y = 0; y < 32; y++ ){
					for ( int x = 0; x < 32; x++ ){
						c.data( 0, x, y, z ) = img[x + (y * 32) + (32 * 32)*z] / 255.f;
					}
				}
			}
			for ( int b = 0; b < 10; ++b ){
				c.out( 0, b, 0, 0 ) = *label == b ? 1.0f : 0.0f;
			}
			cases.push_back( c );
		}
		delete[] buffer;
		return cases;
	}

	vector<CaseObject> readCases(string data_json_path, string mode)
	{
		vector<CaseObject> cases;

		JSONObject *data_json = new JSONObject();
		vector <json_token_t*> data_tokens = data_json->load(data_json_path);
		string dataset_type = data_json->getChildValueForToken(data_tokens[0], "type");   // data_tokens[0] := json root

		if(dataset_type=="mnist"){
			cases= readCasesMNIST(data_json, data_tokens, mode);
		}else if(dataset_type=="cifar10"){
			cases= readCasesCifar10(data_json, data_tokens, mode);
		}

		delete data_json;

		return cases;
	}

};
