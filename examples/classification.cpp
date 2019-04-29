#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include "lightnet.h"

extern vector<CaseObject> readCases(string data_json_path, string model_json_path, string mode); // dataset.cpp

float trainClassification( int step, vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string optimizer, ThreadPool& thread_pool, ParameterObject *parameter_object ){
#ifdef GPU_CUDA
	return trainNetworkGPU( step, layers, data, expected, optimizer, parameter_object);
#else
	return trainNetwork( step, layers, data, expected, optimizer, thread_pool, parameter_object);
#endif
}

float testClassification( vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string optimizer, ThreadPool& thread_pool ){
#ifdef GPU_CUDA
	return testNetworkGPU( layers, data, expected, optimizer);
#else
	return testNetwork( layers, data, expected, optimizer, thread_pool );
#endif
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
	// DatasetObject2 *dataset = new DatasetObject2();
	vector<CaseObject> train_cases = readCases(data_json_path, model_json_path, "train");
	vector<CaseObject> test_cases = readCases(data_json_path, model_json_path, "test");

	printf("\nTrain cases :%lu,  Test cases  :%lu\n\n", train_cases.size(), test_cases.size());
	if(train_cases.size()==0 || test_cases.size()==0){
		exit(0);
	}

	// model
	ParameterObject *parameter_object = new ParameterObject();
	JSONObject *model_json = new JSONObject();
	vector <json_token_t*> model_tokens = model_json->load(model_json_path);
	loadModelParameters(model_json, model_tokens, parameter_object);
	parameter_object->printParameters();

	printf("Start training\n\n");
	CaseObject batch_cases {TensorObject<float>( parameter_object->batch_size, train_cases[0].data.size.x,  train_cases[0].data.size.y,  train_cases[0].data.size.z ), TensorObject<float>( parameter_object->batch_size, 10, 1, 1 )};

	vector<LayerObject*> layers = loadModel(model_json, model_tokens, batch_cases, parameter_object->learning_rate, parameter_object->weights_decay, parameter_object->momentum);
	printf("\n");
	long step = 0;
	if(checkpoint_path.length()>0){
		step = loadLayersWeights( layers, checkpoint_path );
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

	ThreadPool thread_pool(parameter_object->threads);

	while( step < 1000000 ){
		int randi = rand() % (train_cases.size()-parameter_object->batch_size);
		for( unsigned j = randi; j < (randi+parameter_object->batch_size); ++j ){
			t = train_cases[j];
			unsigned batch_index_in = (j-randi)*data_size;
			unsigned batch_index_out = (j-randi)*out_size;
			memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, data_float_size );
			memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, out_float_size );
		}

		float train_err = trainClassification( step, layers, batch_cases.data, batch_cases.out, parameter_object->optimizer, thread_pool, parameter_object );
		train_amse += train_err;
		train_increment++;
		step++;

		if (step % parameter_object->save_span == 0){
			string filename        = "checkpoints/" + data_model_name + "_" + to_string(step) + ".model";
			cout << "Saving weights to " << filename << " ..." << endl;
			saveLayersWeights(step, layers, filename);
		}
		if (step % parameter_object->save_latest_span == 0){
			string filename        = "checkpoints/" + data_model_name + "_latest.model";
			saveLayersWeights(step, layers, filename);
		}

		if ( step % parameter_object->train_output_span == 0 ){
			auto finish = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = finish - start;
			cout << "step " << step << endl;
			cout << "  train error=" << train_amse/train_increment << ", Elapsed time: " << elapsed.count() << " s\n";
			start = finish;

			randi = rand() % (test_cases.size()-parameter_object->batch_size);
			for( unsigned j = randi; j < (randi+parameter_object->batch_size); ++j ){
				CaseObject t = test_cases[j];
				unsigned batch_index_in = (j-randi)*(data_size);
				unsigned batch_index_out = (j-randi)*(out_size);
				memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, data_float_size );
				memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, out_float_size );
			}

			float test_err = testClassification( layers, batch_cases.data, batch_cases.out, parameter_object->optimizer, thread_pool );
			test_amse += test_err;
			test_increment++;
			cout << "  test error =" << test_amse/test_increment << "\n";
		}
	}
}
