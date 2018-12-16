#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "lightnet.h"
#include "TensorObject.h"

using namespace std;

uint32_t byteswap_uint32(uint32_t a)
{
	return ((((a >> 24) & 0xff) << 0) |
		(((a >> 16) & 0xff) << 8) |
		(((a >> 8) & 0xff) << 16) |
		(((a >> 0) & 0xff) << 24));
}

float trainMNIST( vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, bool is_print, float learning_rate, string opt){
	// forward
	for( int i = 0; i < layers.size(); i++ ){
		// printf("----layer %d----\n", i);
		if( i == 0 ){
			activate( layers[i], data );
		}else{
			activate( layers[i], layers[i-1]->out );
		}
	}
	// backward
	TensorObject<float> grads = layers.back()->out - expected;
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

	if(opt=="mse"){
		float err = 0;
		for ( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++ ){
			float f = expected.data[i];
			if ( f > 0.5 ){
				err += abs(grads.data[i]);
			}
		}
		return err * 100;
	}else{
		float loss = 0.0;
		for ( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++ ){
	    loss += (-expected.data[i] * log(layers.back()->out.data[i]));
	  }
		if(is_print){
			printf("----GT----\n");
			print_tensor(expected);
			printf("----output----\n");
			print_tensor(layers.back()->out);
			for(int i = layers.size() - 1; i >= 1; i-- ){
				printf(" ----layer %d ----\n",i);
				print_tensor(layers[i]->grads_in);
			}
		}
		return loss;
	}
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

vector<CaseObject> read_test_cases(string data_json_path)
{
	JSONObject *data_json = new JSONObject();
	std::vector <json_token_t*> data_tokens = data_json->load(data_json_path);
	string train_dir = data_json->getChildValueForToken(data_tokens[0], "train_dir");   // data_tokens[0] := json root

	vector<CaseObject> cases;
	uint8_t* train_image = read_file( (train_dir + "train-images-idx3-ubyte").c_str() );
	uint8_t* train_labels = read_file( (train_dir + "train-labels-idx1-ubyte").c_str() );
	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	for (int i=0; i<case_count; i++){
		CaseObject c {TensorObject<float>( 28, 28, 1 ), TensorObject<float>( 10, 1, 1 )};
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

	vector<CaseObject> cases = read_test_cases(data_json_path);
	JSONObject *model_json = new JSONObject();
	std::vector <json_token_t*> model_tokens = model_json->load(model_json_path);

	// neural network
	json_token_t* nueral_network = model_json->getChildForToken(model_tokens[0], "net");
	float learning_rate = std::stof( model_json->getChildValueForToken(nueral_network, "learning_rate") );
	string opt = model_json->getChildValueForToken(nueral_network, "optimization");

	float amse = 0;
	int ic = 0;
	int BATCH_SIZE = 64;

	vector<LayerObject*> layers = loadModel(model_json, model_tokens, cases, learning_rate);

	printf("Start training data:%lu \n", cases.size());
	for ( long ep = 0; ep < 1000000; ){
		int randindx = rand() % (cases.size()-BATCH_SIZE);
		for (unsigned j = randindx; j < (randindx+BATCH_SIZE); ++j){
			CaseObject t = cases[j];

			bool is_print = false;
			if ( (ep+1) % 1000 == 0 ){
				is_print = true;
			}
			float xerr = trainMNIST( layers, t.data, t.out, is_print, learning_rate, opt);
			if(-1<xerr && xerr<10000){
				amse += xerr;
			}
			ic++;
			ep++;

			if ( ep % 1000 == 0 ){
				cout << "case " << ep << " err=" << amse/ic << endl;
			}
		}
	}

}
