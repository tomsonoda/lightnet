#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <vector>
#include "lightnet.h"
#include "ImageProcessor.h"

using namespace std;

/*
void getDataList(std::string dir_string, std::vector<string> &image_paths, std::vector<string> &label_paths)
{
  Utils *utils = new Utils();
  if (utils->stringEndsWith(dir_string, "/")==0){
    dir_string = dir_string + "/";
  }
  std::string image_dir_string = dir_string + "images/";
  std::string label_dir_string = dir_string + "labels/";
  std::string image_ext = "jpg";
  std::string label_ext = "txt";

  vector<string> files = vector<string>();
  utils->listDir(image_dir_string,files,image_ext);
  for (int j=0; j<files.size(); ++j){
    std::string image_path = files[j];
    std::string label_path = files[j];
    image_path = image_dir_string + image_path;
    label_path = label_dir_string + utils->stringReplace(label_path, image_ext, label_ext);

    std::ifstream ifs(label_path);
    if(ifs.is_open()){ // if exists
      cout << image_path << endl;
      cout << label_path << endl;
      image_paths.push_back(image_path);
    }else{
      cout << "Not exists:" + label_path << endl;
    }
  }
  delete utils;
}

float train( vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected)
{
  TensorObject<float> grads = layers.back()->out - expected;
  ThreadPool thread_pool(4);
  for (int i=0; i<layers.size(); ++i){
    if (i== 0){
      forward(layers[i], data, thread_pool );
    }else{
      forward(layers[i], layers[i-1]->out, thread_pool );
    }
  }
  for (int i=layers.size()-1; i>=0; i--){
    if (i==layers.size()-1){
      backward(layers[i], grads, thread_pool );
    }else{
      backward(layers[i], layers[i+1]->dz, thread_pool );
    }
  }

  for(int i=0; i<layers.size(); ++i){
    update_weights(layers[i]);
  }

  float err = 0;
  print_tensor(grads);
  for( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; ++i ){
    float f = expected.data[i];
    if ( f > 0.5 ){
      err += abs(grads.data[i]);
    }
  }
  return err * 100;
}

vector<CaseObject> readCases(std::string data_config_path)
{
  JSONObject *data_json = new JSONObject();
  std::vector <json_token_t*> data_tokens = data_json->load(data_config_path);
  printf("- train_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "train_dir")).c_str());   // data_tokens[0] := json root
  printf("- checkpoint_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "checkpoint_dir")).c_str());
  printf("- test_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "test_dir")).c_str());
  printf("- test_results_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "test_results_dir")).c_str());

  std::vector<string> train_image_paths;
  std::vector<string> train_label_paths;
  getDataList(data_json->getChildValueForToken(data_tokens[0], "train_dir"), train_image_paths, train_label_paths);
  ImageProcessor *image_processor = new ImageProcessor();
  for(int i=0; i<train_image_paths.size(); ++i){
    image_st image = image_processor->readImageFile(train_image_paths[i], 416, 416, 3);
    printf("Read image: %s %d %d\n", train_image_paths[i].c_str(), image.width, image.height);
    // image_processor->writeImageFilePNG(utils->stringReplace(train_image_paths[i], "jpg", "png"), image);
  }
  vector<CaseObject> cases;
  return cases;
}

void objectDetection(int argc, char **argv)
{
  if(argc < 6){
    fprintf(stderr, "usage: %s %s <train|test> <data_config_path> <model_config_path> <model_path> [other options...]\n", argv[0], argv[1]);
    return;
  }
	string data_config_path = argv[3];
  string model_config_path = argv[4];
  string model_path = argv[5];

  ParameterObject *argument_processor = new ParameterObject();
  string classes_path = argument_processor->getCharsParameter(argc, argv, "-classes_path", "classes.txt");
  float threshold = argument_processor->getFloatParameter(argc, argv, "-threshold", .3);
  delete argument_processor;

  if(strcmp(argv[2], "train")==0){
    if(argc>5){
      trainObjectDetection(model_config_path, model_path, data_config_path);
    }else{
      fprintf(stderr, "usage: %s %s train <model_config_path> <model_path> <data_config_path> [other options]\n", argv[0], argv[1]);
      return;
    }
  }else if(strcmp(argv[2], "test")==0){
    if(argc>5){
      testObjectDetection(model_config_path, model_path, classes_path, data_config_path, threshold);
    }else{
      fprintf(stderr, "usage: %s %s test <model_config_path> <model_path> <data_config_path> [other options]\n", argv[0], argv[1]);
      return;
    }
  }
}
*/


float trainObjectDetection( int step, vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string optimizer, ThreadPool& thread_pool, ParameterObject *parameter_object ){

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

		if ( step % parameter_object->train_output_span == 0 ){
			printf("----GT----\n");
			print_tensor(expected);
			printf("----output----\n");
			print_tensor(layers[layers.size()-1]->out);
		}
		return loss;
	}
}

float testObjectDetection( vector<LayerObject*>& layers, TensorObject<float>& data, TensorObject<float>& expected, string optimizer, ThreadPool& thread_pool ){

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
	// dataset
	DatasetObject *dataset = new DatasetObject();
	vector<CaseObject> train_cases = dataset->readCases(data_json_path, "train");
	vector<CaseObject> test_cases = dataset->readCases(data_json_path, "test");

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

		float train_err = trainObjectDetection( step, layers, batch_cases.data, batch_cases.out, parameter_object->optimizer, thread_pool, parameter_object );
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

			float test_err = testObjectDetection( layers, batch_cases.data, batch_cases.out, parameter_object->optimizer, thread_pool );
			test_amse += test_err;
			test_increment++;
			cout << "  test error =" << test_amse/test_increment << "\n";
		}
	}
}
