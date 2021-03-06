#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include "lightnet.h"

extern vector<CaseObject> readCases(string data_json_path, string model_json_path, string mode); // dataset.cpp

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
	delete utils;

	// dataset
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

#ifdef GPU_CUDA
	printf("Start training [GPU]\n\n");
#else
	printf("Start training\n\n");
#endif

	CaseObject batch_cases {TensorObject<float>( parameter_object->batch_size, train_cases[0].data.size.x,  train_cases[0].data.size.y,  train_cases[0].data.size.z ), TensorObject<float>( parameter_object->batch_size, 10, 1, 1 )};

	vector<LayerObject*> layers = loadModel(model_json, model_tokens, batch_cases, parameter_object->learning_rate, parameter_object->weights_decay, parameter_object->momentum);
	printf("\n");
	long step = 0;

	if(checkpoint_path.length()>0){
		cout << checkpoint_path << endl;
		step = loadLayersWeights( layers, checkpoint_path );
	}

	auto start = std::chrono::high_resolution_clock::now();
	CaseObject t = train_cases[0];

	int data_size = t.data.size.x * t.data.size.y * t.data.size.z;
	int out_size = t.out.size.x * t.out.size.y * t.out.size.z;
	int data_float_size = t.data.size.x * t.data.size.y * t.data.size.z * sizeof(float);
	int out_float_size = t.out.size.x * t.out.size.y * t.out.size.z * sizeof(float);

	ThreadPool thread_pool(parameter_object->threads);

#ifdef GPU_CUDA

	std::vector<float *> outputArrays;
	std::vector<float *> dzArrays;
	std::vector<float *> dzInArrays;

	for( unsigned int i = 0; i < (layers.size()); ++i ){
		int o_size = layers[i]->out.size.b * layers[i]->out.size.x * layers[i]->out.size.y * layers[i]->out.size.z;
		float *gpu_layer_output_array = gpu_cuda::cudaMakeArray( NULL, o_size );
		outputArrays.push_back(gpu_layer_output_array);

		int dz_size = layers[i]->dz.size.b * layers[i]->dz.size.x * layers[i]->dz.size.y * layers[i]->dz.size.z;
		float *gpu_layer_dz_array = gpu_cuda::cudaMakeArray( NULL, dz_size );
		dzArrays.push_back(gpu_layer_dz_array);

		int dz_in_size = layers[i]->dz_in.size.b * layers[i]->dz_in.size.x * layers[i]->dz_in.size.y * layers[i]->dz_in.size.z;
		float *gpu_layer_dz_in_array = gpu_cuda::cudaMakeArray( NULL, dz_in_size );
		dzInArrays.push_back(gpu_layer_dz_in_array);
	}

	float *gpu_in_array = gpu_cuda::cudaMakeArray( NULL, parameter_object->batch_size * data_size );
	float *gpu_out_array = gpu_cuda::cudaMakeArray( NULL, parameter_object->batch_size * out_size );

#endif

	while( step < 10000 ){

		int randi = rand() % (train_cases.size()-parameter_object->batch_size);
		for( unsigned j = randi; j < (randi+parameter_object->batch_size); ++j ){
			CaseObject tt = train_cases[j];
			unsigned batch_index_in = (j-randi)*data_size;
			unsigned batch_index_out = (j-randi)*out_size;
			memcpy( &(batch_cases.data.data[batch_index_in]), tt.data.data, data_float_size );
			memcpy( &(batch_cases.out.data[batch_index_out]), tt.out.data, out_float_size );
		}

		float train_err = 0.0;
#ifdef GPU_CUDA
		train_err = trainNetworkGPU( step, layers, batch_cases.data, batch_cases.out, parameter_object, outputArrays, dzArrays, dzInArrays, gpu_in_array, gpu_out_array );
#else
		train_err = trainNetwork( step, layers, batch_cases.data, batch_cases.out, parameter_object, thread_pool );
#endif

	if ( step % parameter_object->train_output_span == 0 ){
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		cout << "step " << step << endl;
		cout << "  train error=" << train_err << ", Elapsed time: " << elapsed.count() << " s\n";
		start = finish;

		randi = rand() % (test_cases.size()-parameter_object->batch_size);
		for( unsigned j = randi; j < (randi+parameter_object->batch_size); ++j ){
			CaseObject t = test_cases[j];
			unsigned batch_index_in = (j-randi)*(data_size);
			unsigned batch_index_out = (j-randi)*(out_size);
			memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, data_float_size );
			memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, out_float_size );
		}

		// float test_err = testClassification( layers, batch_cases.data, batch_cases.out, parameter_object->loss_function, thread_pool, outputArrays, dzArrays );
	#ifdef GPU_CUDA
		float test_err = testNetworkGPU( layers, batch_cases.data, batch_cases.out, parameter_object->loss_function, outputArrays, dzArrays);
	#else
		float test_err = testNetwork( layers, batch_cases.data, batch_cases.out, parameter_object->loss_function, thread_pool);
	#endif

		cout << "  test error =" << test_err << "\n";
	}

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

	}

}
