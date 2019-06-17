#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include "lightnet.h"

using namespace std;

// in dataset.cpp
extern vector<CasePaths> listImageLabelCasePaths( JSONObject *data_json, vector<json_token_t*> data_tokens, JSONObject *model_json, vector<json_token_t*> model_tokens, string mode );
extern CaseObject readImageLabelCase( CasePaths case_paths, JSONObject *model_json, vector<json_token_t*> model_tokens );
extern float boxTensorIOU(TensorObject<float> &t_a, TensorObject<float> &t_b, JSONObject *model_json, vector<json_token_t*> model_tokens );

float trainObjectDetection( int step, vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string optimizer, ThreadPool& thread_pool, ParameterObject *parameter_object, vector<float *>& outputArrays )
{
#ifdef GPU_CUDA
	return trainNetworkGPU( step, layers, data, expected, optimizer, parameter_object, outputArrays );
#else
	return trainNetwork( step, layers, data, expected, optimizer, thread_pool, parameter_object);
#endif
}

float testObjectDetection( vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string optimizer, ThreadPool& thread_pool, JSONObject *model_json, vector<json_token_t*> model_tokens, ParameterObject *parameter_object ){
	float loss_value = 0.0;
#ifdef GPU_CUDA
	loss_value = testNetworkGPU( layers, data, expected, optimizer );
#else
	loss_value = testNetwork( layers, data, expected, optimizer, thread_pool );
#endif

	float best_iou = 0.0;
	TensorObject<float> output_tensor = layers.back()->out;
	int out_size = output_tensor.size.x * output_tensor.size.y * output_tensor.size.z;
	int out_float_size = output_tensor.size.x * output_tensor.size.y * output_tensor.size.z * sizeof(float);

	for( int b=0; b<output_tensor.size.b; ++b ){
		TensorObject<float> output { 1, output_tensor.size.x, output_tensor.size.y, output_tensor.size.z };
		TensorObject<float> truth { 1, expected.size.x, expected.size.y, expected.size.z };
		int batch_index = b * out_size;
		memcpy( output.data, &(output_tensor.data[batch_index]), out_float_size );
		memcpy( truth.data, &(expected.data[batch_index]), out_float_size );
		float iou = boxTensorIOU(output, truth, model_json, model_tokens);
		if (iou > best_iou){
			best_iou = iou;
		}
	}
	cout << "  test IOU : " << best_iou << endl;

	return loss_value;
}

void objectDetection(int argc, char **argv)
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
	JSONObject *data_json = new JSONObject();
	JSONObject *model_json = new JSONObject();
	vector <json_token_t*> data_tokens = data_json->load(data_json_path);
	vector <json_token_t*> model_tokens = model_json->load(model_json_path);

	// dataset
	vector<CasePaths> train_case_paths = listImageLabelCasePaths( data_json, data_tokens, model_json, model_tokens, "train" );
	vector<CasePaths> test_case_paths = listImageLabelCasePaths( data_json, data_tokens, model_json, model_tokens, "test");
	printf("\nTrain cases :%lu,  Test cases  :%lu\n\n", train_case_paths.size(), test_case_paths.size());
	if(train_case_paths.size()==0 || test_case_paths.size()==0){
		exit(0);
	}

	// model
	ParameterObject *parameter_object = new ParameterObject();
	loadModelParameters(model_json, model_tokens, parameter_object);
	parameter_object->printParameters();
	printf("Start training\n\n");
	CaseObject t = readImageLabelCase( train_case_paths[0], model_json, model_tokens );
	CaseObject batch_cases {
		TensorObject<float>( parameter_object->batch_size, t.data.size.x,  t.data.size.y,  t.data.size.z ),
		TensorObject<float>( parameter_object->batch_size, t.out.size.x,  t.out.size.y,  t.out.size.z )
	};

	vector<LayerObject*> layers = loadModel(model_json, model_tokens, batch_cases, parameter_object->learning_rate, parameter_object->weights_decay, parameter_object->momentum);
	printf("\n");
	long step = 0;
	if(checkpoint_path.length()>0){
		step = loadLayersWeights( layers, checkpoint_path );
	}

	auto start = std::chrono::high_resolution_clock::now();

	int data_size = t.data.size.x * t.data.size.y * t.data.size.z;
	int out_size = t.out.size.x * t.out.size.y * t.out.size.z;
	int data_float_size = t.data.size.x * t.data.size.y * t.data.size.z * sizeof(float);
	int out_float_size = t.out.size.x * t.out.size.y * t.out.size.z * sizeof(float);

	ThreadPool thread_pool(parameter_object->threads);

	std::vector<float *> outputArrays;
	
#ifdef GPU_CUDA
	for( unsigned int i = 0; i < (layers.size()); ++i ){
		int o_size = layers[i]->out.size.b * layers[i]->out.size.x * layers[i]->out.size.y * layers[i]->out.size.z;
		float *gpu_layer_output_array = gpu_cuda::cudaMakeArray( NULL, o_size );
		layers[i]->gpu_out = gpu_layer_output_array;
		outputArrays.push_back(gpu_layer_output_array);
	}
#endif

	while( step < 1000000 ){
		int randi = 1;
		if( (train_case_paths.size()-parameter_object->batch_size) > 0){
			randi = rand() % (train_case_paths.size()-parameter_object->batch_size);
		}
		for( unsigned j = randi; j < (randi+parameter_object->batch_size); ++j ){
			CaseObject t = readImageLabelCase( train_case_paths[j], model_json, model_tokens );
			unsigned batch_index_in = (j-randi)*data_size;
			unsigned batch_index_out = (j-randi)*out_size;
			memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, data_float_size );
			memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, out_float_size );
		}

		// printTensor(batch_cases.out);

		float train_err = trainObjectDetection( step, layers, batch_cases.data, batch_cases.out, parameter_object->optimizer, thread_pool, parameter_object, outputArrays);
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
			cout << "  train error=" << train_err << ", Elapsed time: " << elapsed.count() << " s\n";
			start = finish;

			if( test_case_paths.size()-parameter_object->batch_size >0 ){
				randi = rand() % (test_case_paths.size()-parameter_object->batch_size);
			}else{
				randi = rand();
			}
			for( unsigned j = randi; j < (randi+parameter_object->batch_size); ++j ){
				CaseObject t = readImageLabelCase( test_case_paths[j], model_json, model_tokens );

				unsigned batch_index_in = (j-randi)*(data_size);
				unsigned batch_index_out = (j-randi)*(out_size);
				memcpy( &(batch_cases.data.data[batch_index_in]), t.data.data, data_float_size );
				memcpy( &(batch_cases.out.data[batch_index_out]), t.out.data, out_float_size );
			}

			float test_err = testObjectDetection( layers, batch_cases.data, batch_cases.out, parameter_object->optimizer, thread_pool, model_json, model_tokens, parameter_object);
			cout << "  test error =" << test_err << "\n";
		}
	}

}
