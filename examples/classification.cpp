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

unsigned batch_size = 1;
float learning_rate = 0.01;
float momentum = 0.6;
float weights_decay = 0.01;
string opt = "mse";
int train_output_span = 1000;

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

		if ( step % train_output_span == 0 ){
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

void loadModelParameters(JSONObject *model_json, vector <json_token_t*> model_tokens)
{
	json_token_t* nueral_network = model_json->getChildForToken(model_tokens[0], "net");

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
	tmp = model_json->getChildValueForToken(nueral_network, "weights_decay");
	if(tmp.length()>0){
		weights_decay = std::stof( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "optimization");
	if(tmp.length()>0){
		opt = tmp;
	}
	tmp = model_json->getChildValueForToken(nueral_network, "train_output_span");
	if(tmp.length()>0){
		train_output_span = std::stoi( tmp );
	}

	if(batch_size<0){
		fprintf(stderr, "Batch size should be 1>=.");
		exit(0);
	}
}

void classification(int argc, char **argv)
{
	cout << endl;

	string data_json_path = argv[2];
	string model_json_path = argv[3];

	// dataset
	DatasetObject *dataset = new DatasetObject();
	vector<CaseObject> train_cases = dataset->readCases(data_json_path, "train");
	vector<CaseObject> test_cases = dataset->readCases(data_json_path, "test");
	printf("Train cases :%lu,  Test cases  :%lu\n\n", train_cases.size(), test_cases.size());
	if(train_cases.size()==0 || test_cases.size()==0){
		exit(0);
	}

	// model
	JSONObject *model_json = new JSONObject();
	vector <json_token_t*> model_tokens = model_json->load(model_json_path);
	loadModelParameters(model_json, model_tokens);

	float amse = 0;
	float test_amse = 0;
	int ic = 0;
	int test_ic = 0;
	printf("Start training - batch_size:%d, learning_rate:%f, momentum:%f, weights_decay:%f, optimizer:%s\n\n", batch_size, learning_rate, momentum, weights_decay, opt.c_str());

	CaseObject batch_cases {TensorObject<float>( batch_size, train_cases[0].data.size.x,  train_cases[0].data.size.y,  train_cases[0].data.size.z ), TensorObject<float>( batch_size, 10, 1, 1 )};
	vector<LayerObject*> layers = loadModel(model_json, model_tokens, batch_cases, learning_rate, weights_decay, momentum);
	printf("\n");

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

		if ( step % train_output_span == 0 ){
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
