#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "lightnet.hpp"

using namespace std;

uint32_t byteswap_uint32(uint32_t a)
{
	return ((((a >> 24) & 0xff) << 0) |
		(((a >> 16) & 0xff) << 8) |
		(((a >> 8) & 0xff) << 16) |
		(((a >> 0) & 0xff) << 24));
}

float trainMNIST( vector<layer_t*>& layers, tensor_t<float>& data, tensor_t<float>& expected){
	// forward
	for( int i = 0; i < layers.size(); i++ ){
		if( i == 0 ){
			activate( layers[i], data );
		}else{
			activate( layers[i], layers[i-1]->out );
		}
	}

	// backward
	tensor_t<float> grads = layers.back()->out - expected;
	for ( int i = layers.size() - 1; i >= 0; i-- ){
		if ( i == layers.size() - 1 ){
			calc_grads( layers[i], grads );
		}else{
			calc_grads( layers[i], layers[i+1]->grads_in );
		}
	}

	// update
	for ( int i = 0; i < layers.size(); i++ ){
		fix_weights( layers[i] );
	}

	float err = 0;
	for ( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++ ){
		float f = expected.data[i];
		if ( f > 0.5 ){
			err += abs(grads.data[i]);
		}
	}
	return err * 100;
}

uint8_t* read_file( const char* szFile ){
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

vector<case_t> read_test_cases(string data_json_path)
{
	printf("read_test_cases: %s\n", data_json_path.c_str());
	JSONObject *data_json = new JSONObject();
	std::vector <json_token_t*> data_tokens = data_json->load(data_json_path);
	string train_dir = data_json->getChildValueForToken(data_tokens[0], "train_dir");   // data_tokens[0] := json root

	vector<case_t> cases;
	uint8_t* train_image = read_file( (train_dir + "train-images-idx3-ubyte").c_str() );
	uint8_t* train_labels = read_file( (train_dir + "train-labels-idx1-ubyte").c_str() );
	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );
	printf("case_count=%d\n", case_count);

	for (int i=0; i<case_count; i++){
		case_t c {tensor_t<float>( 28, 28, 1 ), tensor_t<float>( 10, 1, 1 )};
		uint8_t* img = train_image + 16 + i * (28 * 28);
		uint8_t* label = train_labels + 8 + i;
		for ( int x = 0; x < 28; x++ ){
			for ( int y = 0; y < 28; y++ ){
				c.data( x, y, 0 ) = img[x + y * 28] / 255.f;
			}
		}
		for ( int b = 0; b < 10; b++ ){
			c.out( b, 0, 0 ) = *label == b ? 1.0f : 0.0f;
		}
		cases.push_back( c );
	}
	delete[] train_image;
	delete[] train_labels;
	return cases;
}

void mnist(int argc, char **argv)
{
	string data_json_path = argv[2];
	string model_json_path = argv[3];

	vector<case_t> cases = read_test_cases(data_json_path);
	vector<layer_t*> layers = loadModel(model_json_path, cases);

	float amse = 0;
	int ic = 0;

	for ( long ep = 0; ep < 100000; ){
		for ( case_t& t : cases ){
			float xerr = trainMNIST( layers, t.data, t.out );
			amse += xerr;
			ep++;
			ic++;
			if ( ep % 1000 == 0 ){
				cout << "case " << ep << " err=" << amse/ic << endl;
			}
		}
	}
	// end:

	while ( true ){
		uint8_t * data = read_file( "test.ppm" );

		if ( data ){
			uint8_t * usable = data;
			while ( *(uint32_t*)usable != 0x0A353532 ){
				usable++;
			}

#pragma pack(push, 1)
			struct RGB
			{
				uint8_t r, g, b;
			};
#pragma pack(pop)

			RGB * rgb = (RGB*)usable;
			tensor_t<float> image(28, 28, 1);
			for ( int i = 0; i < 28; i++ ){
				for ( int j = 0; j < 28; j++ ){
					RGB rgb_ij = rgb[i * 28 + j];
					image( j, i, 0 ) = (((float)rgb_ij.r
							     + rgb_ij.g
							     + rgb_ij.b)
							    / (3.0f*255.f));
				}
			}

			for (int i=0; i<layers.size(); i++){
				if (i== 0){
					activate(layers[i], image);
				}else{
					activate(layers[i], layers[i-1]->out);
				}
			}

			tensor_t<float>& out = layers.back()->out;
			for ( int i = 0; i < 10; i++ ){
				printf( "[%i] %f\n", i, out( i, 0, 0 )*100.0f );
			}
			delete[] data;
		}

		struct timespec wait;
		wait.tv_sec = 1;
		wait.tv_nsec = 0;
		nanosleep(&wait, nullptr);
	}
}
