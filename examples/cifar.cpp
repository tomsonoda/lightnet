#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include "lightnet.h"
#include "Utils.h"

using namespace std;

#define OUTPUT_TIMING 20

float trainCifar( int step, vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string opt ){

	for( int i = 0; i < layers.size(); i++ ){
		if( i == 0 ){
			forward( layers[i], data );
		}else{
			forward( layers[i], layers[i-1]->out );
		}
	}

	TensorObject<float> grads = layers.back()->out - expected;
	for( int i = 0; i < layers.size(); i++ ){
		layers[i]->dz_in.clear();
		layers[i]->dz.clear();
	}

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

		if ( step % OUTPUT_TIMING == 0 ){
			printf("----GT----\n");
			print_tensor(expected);
			printf("----output----\n");
			print_tensor(layers[layers.size()-1]->out);
		}
		return loss;
	}
}

float testCifar( vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string opt ){

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

vector<CaseObject> read_cifar_cases(string data_json_path, string mode)
{
	JSONObject *data_json = new JSONObject();
	DatasetObject *dataset = new DatasetObject();
	std::vector <json_token_t*> data_tokens = data_json->load(data_json_path);
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

	for (int i=0; i<case_count; i++){
		CaseObject c {TensorObject<float>( 1, 32, 32, 3 ), TensorObject<float>( 1, 10, 1, 1 )};
		uint8_t* img = buffer + i * (32 * 32 * 3 + 1) + 1;
		uint8_t* label = buffer + i * (32 * 32 * 3 + 1);
		for(int z = 0; z < 3; z++ ){
			for ( int y = 0; y < 32; y++ ){
				for ( int x = 0; x < 32; x++ ){
					c.data( 0, x, y, z ) = img[x + (y * 32) + (32 * 32)*z] / 255.f;
				}
			}
		}
		for ( int b = 0; b < 10; b++ ){
			c.out( 0, b, 0, 0 ) = *label == b ? 1.0f : 0.0f;
		}
		cases.push_back( c );
	}

	delete[] buffer;

	return cases;
}

void cifar(int argc, char **argv)
{
	cout << endl;

	string data_json_path = argv[2];
	string model_json_path = argv[3];

	vector<CaseObject> train_cases = read_cifar_cases(data_json_path, "train");
	vector<CaseObject> test_cases = read_cifar_cases(data_json_path, "test");
	JSONObject *model_json = new JSONObject();
	std::vector <json_token_t*> model_tokens = model_json->load(model_json_path);

	json_token_t* nueral_network = model_json->getChildForToken(model_tokens[0], "net");

	unsigned batch_size = 1;
	float learning_rate = 0.01;
	float momentum = 0.6;
	float decay = 0.01;
	string opt = "mse";

	string tmp = model_json->getChildValueForToken(nueral_network, "batch_size");
	if(tmp.length()>0){
		batch_size = std::stoi( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "learning_rate");
	if(tmp.length()>0){
		learning_rate = std::stof( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "momentum");
	if(tmp.length()>0){
		momentum = std::stof( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "decay");
	if(tmp.length()>0){
		decay = std::stof( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "optimization");
	if(tmp.length()>0){
		opt = tmp;
	}

	if(batch_size<0){
		fprintf(stderr, "Batch size should be 1>=.");
		exit(0);
	}

	float amse = 0;
	float test_amse = 0;
	int ic = 0;
	int test_ic = 0;

	CaseObject batch_cases {TensorObject<float>( batch_size, 32, 32, 3 ), TensorObject<float>( batch_size, 10, 1, 1 )};
	vector<LayerObject*> layers = loadModel(model_json, model_tokens, batch_cases, learning_rate, decay, momentum);
	printf("\n\nbatch_size:%d, learning_rate:%f, momentum:%f, decay:%f, optimizer:%s\n\n", batch_size, learning_rate, momentum, decay, opt.c_str());
	printf("Start training :%lu \n\n", train_cases.size());
	printf("Test cases :%lu \n\n", test_cases.size());

	if(train_cases.size()==0 || test_cases.size()==0){
		exit(0);
	}

	auto start = std::chrono::high_resolution_clock::now();
	for( long step = 0; step < 1000000; ){
		int randi = rand() % (train_cases.size()-batch_size);
		for( unsigned j = randi; j < (randi+batch_size); j++ ){
			CaseObject t = train_cases[j];
			unsigned batch_index_in = (j-randi)*(t.data.size.x * t.data.size.y * t.data.size.z);
			unsigned batch_index_out = (j-randi)*(t.out.size.x * t.out.size.y * t.out.size.z);
			memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, (t.data.size.x * t.data.size.y * t.data.size.z) * sizeof(float) );
			memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, (t.out.size.x * t.out.size.y * t.out.size.z) * sizeof(float) );
		}

		float xerr = trainCifar( step, layers, batch_cases.data, batch_cases.out, opt );
		amse += xerr;
		ic++;
		step++;

		if ( step % OUTPUT_TIMING == 0 ){
			auto finish = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = finish - start;
			cout << "step " << step << endl;
			cout << "  train_err=" << amse/ic << ", Elapsed time: " << elapsed.count() << " s\n";
			start = finish;

			randi = rand() % (test_cases.size()-batch_size);
			for( unsigned j = randi; j < (randi+batch_size); j++ ){
				CaseObject t = test_cases[j];
				unsigned batch_index_in = (j-randi)*(t.data.size.x * t.data.size.y * t.data.size.z);
				unsigned batch_index_out = (j-randi)*(t.out.size.x * t.out.size.y * t.out.size.z);
				memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, (t.data.size.x * t.data.size.y * t.data.size.z) * sizeof(float) );
				memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, (t.out.size.x * t.out.size.y * t.out.size.z) * sizeof(float) );
			}

			float test_err = testCifar( layers, batch_cases.data, batch_cases.out, opt );
			test_amse += test_err;
			test_ic++;
			cout << "  test_err =" << test_amse/test_ic << "\n";
		}
	}
}
