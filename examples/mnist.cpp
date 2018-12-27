#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

#include "lightnet.h"
#include "TensorObject.h"
#include "Utils.h"

using namespace std;

uint32_t revert_uint32(uint32_t a)
{
	return ((((a >> 24) & 0xff) << 0) |
		(((a >> 16) & 0xff) << 8) |
		(((a >> 8) & 0xff) << 16) |
		(((a >> 0) & 0xff) << 24));
}

float trainMNIST( int epoch, vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string opt ){
	for( int i = 0; i < layers.size(); i++ ){
		if( i == 0 ){
			forward( layers[i], data );
		}else{
			forward( layers[i], layers[i-1]->out );
		}
	}

	TensorObject<float> grads = layers.back()->out - expected;
	for ( int i = layers.size() - 1; i >= 0; i-- ){
		if ( i == layers.size() - 1 ){
			backward( layers[i], grads );
		}else{
			backward( layers[i], layers[i+1]->dz );
		}
	}

	for ( int i = 0; i < layers.size(); i++ ){
		update_weights( layers[i] );
	}

	if(opt=="mse"){
		float err = 0;
		for ( int i = 0; i < grads.size.b * grads.size.x * grads.size.y * grads.size.z; i++ ){
			float f = expected.data[i];
			if ( f > 0.5 ){
				err += abs(grads.data[i]);
			}
		}
		return (err * 100)/(float)expected.size.b;

	}else{
		float loss = 0.0;
		for ( int i = 0; i < grads.size.b *grads.size.x * grads.size.y * grads.size.z; i++ ){
	    loss += (-expected.data[i] * log(layers.back()->out.data[i]));
	  }
		loss /= (float)expected.size.b;

		if ( epoch % 1000 == 0 ){
			printf("----GT----\n");
			print_tensor(expected);
			printf("----output----\n");
			print_tensor(layers[layers.size()-1]->out);
		}
		return loss;
	}
}

float testMNIST( vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string opt ){
	for( int i = 0; i < layers.size(); i++ ){
		if( i == 0 ){
			forward( layers[i], data );
		}else{
			forward( layers[i], layers[i-1]->out );
		}
	}

	TensorObject<float> grads = layers.back()->out - expected;

	if(opt=="mse"){
		float err = 0;
		for ( int i = 0; i < grads.size.b * grads.size.x * grads.size.y * grads.size.z; i++ ){
			float f = expected.data[i];
			if ( f > 0.5 ){
				err += abs(grads.data[i]);
			}
		}
		return (err * 100)/(float)expected.size.b;
	}else{
		float loss = 0.0;
		for ( int i = 0; i < grads.size.b *grads.size.x * grads.size.y * grads.size.z; i++ ){
	    loss += (-expected.data[i] * log(layers.back()->out.data[i]));
	  }
		loss /= (float)expected.size.b;
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

vector<CaseObject> read_cases(string data_json_path, string mode)
{
	JSONObject *data_json = new JSONObject();
	std::vector <json_token_t*> data_tokens = data_json->load(data_json_path);
	vector<CaseObject> cases;
	uint8_t* images;
	uint8_t* labels;

	if(mode=="train"){
		string train_dir = data_json->getChildValueForToken(data_tokens[0], "train_dir");   // data_tokens[0] := json root
		images = read_file( (train_dir + "train-images-idx3-ubyte").c_str() );
		labels = read_file( (train_dir + "train-labels-idx1-ubyte").c_str() );
	}else{
		string test_dir = data_json->getChildValueForToken(data_tokens[0], "test_dir");   // data_tokens[0] := json root
		images = read_file( (test_dir + "t10k-images-idx3-ubyte").c_str() );
		labels = read_file( (test_dir + "t10k-labels-idx1-ubyte").c_str() );
	}
	uint32_t case_count = revert_uint32( *(uint32_t*)(images + 4) );

	for (int i=0; i<case_count; i++){
		CaseObject c {TensorObject<float>( 1, 28, 28, 1 ), TensorObject<float>( 1, 10, 1, 1 )};
		uint8_t* img = images + 16 + i * (28 * 28);
		uint8_t* label = labels + 8 + i;
		for ( int x = 0; x < 28; x++ ){
			for ( int y = 0; y < 28; y++ ){
				c.data( 0, x, y, 0 ) = img[x + y * 28] / 255.f;
			}
		}
		for ( int b = 0; b < 10; b++ ){
			c.out( 0, b, 0, 0 ) = *label == b ? 1.0f : 0.0f;
		}
		cases.push_back( c );
	}

	delete[] images;
	delete[] labels;
	return cases;
}

void mnist(int argc, char **argv)
{
	cout << endl;

	string data_json_path = argv[2];
	string model_json_path = argv[3];

	vector<CaseObject> train_cases = read_cases(data_json_path, "train");
	vector<CaseObject> test_cases = read_cases(data_json_path, "test");
	JSONObject *model_json = new JSONObject();
	std::vector <json_token_t*> model_tokens = model_json->load(model_json_path);

	// neural network
	json_token_t* nueral_network = model_json->getChildForToken(model_tokens[0], "net");
	unsigned batch_size = std::stoi( model_json->getChildValueForToken(nueral_network, "batch_size") );
	float learning_rate = std::stof( model_json->getChildValueForToken(nueral_network, "learning_rate") );
	float momentum = std::stof( model_json->getChildValueForToken(nueral_network, "momentum") );
	float decay = std::stof( model_json->getChildValueForToken(nueral_network, "decay") );
	string opt = model_json->getChildValueForToken(nueral_network, "optimization");

	if(batch_size<0){
		fprintf(stderr, "Batch size should be 1>=.");
		exit(0);
	}

	float amse = 0;
	float test_amse = 0;
	int ic = 0;
	int test_ic = 0;

	CaseObject batch_cases {TensorObject<float>( batch_size, 28, 28, 1 ), TensorObject<float>( batch_size, 10, 1, 1 )};
	vector<LayerObject*> layers = loadModel(model_json, model_tokens, batch_cases, learning_rate, decay, momentum);
	printf("\nStart training :%lu learning_rate=%f momentum=%f, decay=%f, optimizer=%s\n\n", train_cases.size(), learning_rate, momentum, decay, opt.c_str());

	auto start = std::chrono::high_resolution_clock::now();
	for( long epoch = 0; epoch < 1000000; ){
		int randi = rand() % (train_cases.size()-batch_size);
		for( unsigned j = randi; j < (randi+batch_size); j++ ){
			CaseObject t = train_cases[j];
			unsigned batch_index_in = (j-randi)*(t.data.size.x * t.data.size.y * t.data.size.z);
			unsigned batch_index_out = (j-randi)*(t.out.size.x * t.out.size.y * t.out.size.z);
			memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, (t.data.size.x * t.data.size.y * t.data.size.z) * sizeof(float) );
			memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, (t.out.size.x * t.out.size.y * t.out.size.z) * sizeof(float) );
		}

		float xerr = trainMNIST( epoch, layers, batch_cases.data, batch_cases.out, opt );
		amse += xerr;
		ic++;
		epoch++;

		if ( epoch % 1000 == 0 ){
			auto finish = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = finish - start;
			cout << "case " << epoch << endl;
			cout << "  train_err=" << amse/ic << ", amse=" << amse << ", ic=" << ic << ", Elapsed time: " << elapsed.count() << " s\n";
			start = finish;

			randi = rand() % (test_cases.size()-batch_size);
			for( unsigned j = randi; j < (randi+batch_size); j++ ){
				CaseObject t = test_cases[j];
				unsigned batch_index_in = (j-randi)*(t.data.size.x * t.data.size.y * t.data.size.z);
				unsigned batch_index_out = (j-randi)*(t.out.size.x * t.out.size.y * t.out.size.z);
				memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, (t.data.size.x * t.data.size.y * t.data.size.z) * sizeof(float) );
				memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, (t.out.size.x * t.out.size.y * t.out.size.z) * sizeof(float) );
			}

			float test_err = testMNIST( layers, batch_cases.data, batch_cases.out, opt );
			test_amse += test_err;
			test_ic++;
			cout << "  test_err =" << test_amse/test_ic << "\n";
		}
	}
}
