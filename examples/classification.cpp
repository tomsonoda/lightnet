#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include "lightnet.h"

#define VERSION_MAJOR    (0)
#define VERSION_MINOR    (1)
#define VERSION_REVISION (0)

using namespace std;

int save_latest_span = 100;
int save_span = 2000;

unsigned batch_size = 1;
float learning_rate = 0.01;
float momentum = 0.6;
float weights_decay = 0.01;
string optimizer = "mse";
int train_output_span = 1000;
int threads = 1;

float trainClassification( int step, vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string optimizer, ThreadPool& thread_pool ){

	for( int i = 0; i < layers.size(); ++i ){
		if( i == 0 ){
			forward( layers[i], data, thread_pool );
		}else{
			forward( layers[i], layers[i-1]->out, thread_pool );
		}
	}

	TensorObject<float> grads = layers.back()->out - expected;
	for( int i = 0; i < layers.size(); ++i ){
		layers[i]->dz_in.clear();
		layers[i]->dz.clear();
	}

	for ( int i = layers.size() - 1; i >= 0; i-- ){
		if ( i == layers.size() - 1 ){
			backward( layers[i], grads, thread_pool );
		}else{
			backward( layers[i], layers[i+1]->dz, thread_pool );
		}
	}

	for ( int i = 0; i < layers.size(); ++i ){
		update_weights( layers[i] );
	}

	if(optimizer=="mse"){
		float err = 0;
		for ( int i = 0; i < grads.size.b * grads.size.x * grads.size.y * grads.size.z; ++i ){
			float f = expected.data[i];
			if ( f > 0.5 ){
				err += abs(grads.data[i]);
			}
		}
		return (err * 100)/(float)expected.size.b;

	}else{
		float loss = 0.0;
		for ( int i = 0; i < grads.size.b *grads.size.x * grads.size.y * grads.size.z; ++i ){
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

float testClassification( vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string optimizer, ThreadPool& thread_pool ){

	for( int i = 0; i < layers.size(); ++i ){
		if( i == 0 ){
			forward( layers[i], data, thread_pool );
		}else{
			forward( layers[i], layers[i-1]->out, thread_pool );
		}
	}

	TensorObject<float> grads = layers.back()->out - expected;

	if(optimizer=="mse"){
		float err = 0;
		for ( int i = 0; i < grads.size.b * grads.size.x * grads.size.y * grads.size.z; ++i ){
			float f = expected.data[i];
			if ( f > 0.5 ){
				err += abs(grads.data[i]);
			}
		}
		return (err * 100)/(float)expected.size.b;
	}else{
		float loss = 0.0;
		for ( int i = 0; i < grads.size.b *grads.size.x * grads.size.y * grads.size.z; ++i ){
	    loss += (-expected.data[i] * log(layers.back()->out.data[i]));
	  }
		loss /= (float)expected.size.b;
		return loss;
	}
}

void saveClassificationWeights( long step, vector<LayerObject*>& layers, string filename )
{
	std::ofstream fout( filename.c_str(), std::ios::binary );
	if (fout.fail()){
		std::cerr << "No weights file to save:" << filename << std::endl;
		return;
	}
	char ver_major    = (char)VERSION_MAJOR;
	char ver_minor    = (char)VERSION_MINOR;
	char ver_revision = (char)VERSION_REVISION;

	fout.write(( char * ) &(ver_major), sizeof( char ) );
	fout.write(( char * ) &(ver_minor), sizeof( char ) );
	fout.write(( char * ) &(ver_revision), sizeof( char ) );
	fout.write(( char * ) &(step), sizeof( long ) );

	for( int i = 0; i < layers.size(); ++i ){
		saveWeights( layers[i], fout );
	}
	fout.close();
	// copy file to latest
	// std::ifstream src(filename, std::ios::binary);
	// std::ofstream dst("checkpoints/latest.model", std::ios::binary);
	// dst << src.rdbuf();
}

long loadClassificationWeights( vector<LayerObject*>& layers, string filename )
{
	std::ifstream fin( filename.c_str(), std::ios::binary );

	if (fin.fail()){
		cout << "Error : Could not open a file to load:" << filename << std::endl;
		exit(0);
	}

	cout << "Loading weights from " << filename << " ..." << endl;
	char ver_major    = 0;
	char ver_minor    = 0;
	char ver_revision = 0;
	long step = 0;
	fin.read(( char * ) &(ver_major), sizeof( char ) );
	fin.read(( char * ) &(ver_minor), sizeof( char ) );
	fin.read(( char * ) &(ver_revision), sizeof( char ) );
	fin.read(( char * ) &(step), sizeof( long ) );
	cout << "Model version: " << to_string(ver_major) << "." << to_string(ver_minor) << "." << to_string(ver_revision) << endl;
	if( (ver_major!=VERSION_MAJOR) || (ver_minor!=VERSION_MINOR) || (ver_revision!=VERSION_REVISION) ){
		cout << "Model version is different from this version:" << to_string(VERSION_MAJOR) << "." << to_string(VERSION_MINOR) << "." << to_string(VERSION_REVISION) << endl;
		exit(0);
	}
	cout << "last step: " << to_string(step) << endl;

	for( int i = 0; i < layers.size(); ++i ){
		loadWeights( layers[i], fin );
	}
	fin.close();
	return step;
}

void loadModelParameters(JSONObject *model_json, vector <json_token_t*> model_tokens)
{
	json_token_t* nueral_network = model_json->getChildForToken(model_tokens[0], "net");

	string tmp = model_json->getChildValueForToken(nueral_network, "batch_size");
	if(tmp.length()>0){
		batch_size = std::stoi( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "threads");
	if(tmp.length()>0){
		threads = std::stoi( tmp );
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
		optimizer = tmp;
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
	string data_json_path = argv[2];
	string model_json_path = argv[3];
	string checkpoint_path = "";
	if(argc>=5){
 		checkpoint_path = argv[4];
	}
	Utils *utils = new Utils();

	string data_json_base = data_json_path.substr(data_json_path.find_last_of("/")+1);
	string model_json_base = model_json_path.substr(model_json_path.find_last_of("/")+1);
	string data_model_name = utils->stringReplace(data_json_base, ".json", "") + "-" + utils->stringReplace(model_json_base, ".json", "");
	// dataset
	DatasetObject *dataset = new DatasetObject();
	vector<CaseObject> train_cases = dataset->readCases(data_json_path, "train");
	vector<CaseObject> test_cases = dataset->readCases(data_json_path, "test");

	printf("\nTrain cases :%lu,  Test cases  :%lu\n\n", train_cases.size(), test_cases.size());
	if(train_cases.size()==0 || test_cases.size()==0){
		exit(0);
	}

	// model
	JSONObject *model_json = new JSONObject();
	vector <json_token_t*> model_tokens = model_json->load(model_json_path);
	loadModelParameters(model_json, model_tokens);

	printf("batch_size    : %d\n", batch_size);
	printf("threads       : %d\n", threads);
	printf("learning_rate : %f\n", learning_rate);
	printf("momentum      : %f\n", momentum);
	printf("weights_decay : %f\n", weights_decay);
	printf("optimizer     : %s\n\n", optimizer.c_str());

	printf("Start training\n\n");
	CaseObject batch_cases {TensorObject<float>( batch_size, train_cases[0].data.size.x,  train_cases[0].data.size.y,  train_cases[0].data.size.z ), TensorObject<float>( batch_size, 10, 1, 1 )};

	vector<LayerObject*> layers = loadModel(model_json, model_tokens, batch_cases, learning_rate, weights_decay, momentum);
	printf("\n");
	long step = 0;
	if(checkpoint_path.length()>0){
		step = loadClassificationWeights( layers, checkpoint_path );
	}

	auto start = std::chrono::high_resolution_clock::now();
	CaseObject t = train_cases[0];

	int data_size = t.data.size.x * t.data.size.y * t.data.size.z;
	int out_size = t.out.size.x * t.out.size.y * t.out.size.z;
	int data_float_size = t.data.size.x * t.data.size.y * t.data.size.z * sizeof(float);
	int out_float_size = t.out.size.x * t.out.size.y * t.out.size.z * sizeof(float);
	float train_amse = 0;
	float test_amse = 0;
	int train_increment = 0;
	int test_increment = 0;

	ThreadPool thread_pool(threads);

	while( step < 1000000 ){
		int randi = rand() % (train_cases.size()-batch_size);
		for( unsigned j = randi; j < (randi+batch_size); ++j ){
			t = train_cases[j];
			unsigned batch_index_in = (j-randi)*data_size;
			unsigned batch_index_out = (j-randi)*out_size;
			memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, data_float_size );
			memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, out_float_size );
		}

		float train_err = trainClassification( step, layers, batch_cases.data, batch_cases.out, optimizer, thread_pool );
		train_amse += train_err;
		train_increment++;
		step++;

		if (step % save_span == 0){
			string filename        = "checkpoints/" + data_model_name + "_" + to_string(step) + ".model";
			cout << "Saving weights to " << filename << " ..." << endl;
			saveClassificationWeights(step, layers, filename);
		}
		if (step % save_latest_span == 0){
			string filename        = "checkpoints/" + data_model_name + "_latest.model";
			saveClassificationWeights(step, layers, filename);
		}

		if ( step % train_output_span == 0 ){
			auto finish = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = finish - start;
			cout << "step " << step << endl;
			cout << "  train error=" << train_amse/train_increment << ", Elapsed time: " << elapsed.count() << " s\n";
			start = finish;

			randi = rand() % (test_cases.size()-batch_size);
			for( unsigned j = randi; j < (randi+batch_size); ++j ){
				CaseObject t = test_cases[j];
				unsigned batch_index_in = (j-randi)*(data_size);
				unsigned batch_index_out = (j-randi)*(out_size);
				memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, data_float_size );
				memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, out_float_size );
			}

			float test_err = testClassification( layers, batch_cases.data, batch_cases.out, optimizer, thread_pool );
			test_amse += test_err;
			test_increment++;
			cout << "  test error =" << test_amse/test_increment << "\n";
		}
	}
}
