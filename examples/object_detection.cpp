#include <iostream>
#include <fstream>
#include <vector>
#include "lightnet.h"
#include "ImageProcessor.h"
#include "NeuralNetwork.h"

using namespace std;

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
  for (int j=0; j<files.size(); j++){
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
  for (int i=0; i<layers.size(); i++){
    if (i== 0){
      forward(layers[i], data);
    }else{
      forward(layers[i], layers[i-1]->out);
    }
  }
  TensorObject<float> grads = layers.back()->out - expected;

  for (int i=layers.size()-1; i>=0; i--){
    if (i==layers.size()-1){
      backward(layers[i], grads);
    }else{
      backward(layers[i], layers[i+1]->dz);
    }
  }

  for(int i=0; i<layers.size(); i++){
    update_weights(layers[i]);
  }

  float err = 0;
  print_tensor(grads);
  for( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++ ){
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
  for(int i=0; i<train_image_paths.size(); i++){
    image_st image = image_processor->readImageFile(train_image_paths[i], 416, 416, 3);
    printf("Read image: %s %d %d\n", train_image_paths[i].c_str(), image.width, image.height);
    // image_processor->writeImageFilePNG(utils->stringReplace(train_image_paths[i], "jpg", "png"), image);
  }
  vector<CaseObject> cases;
  return cases;
}

void trainObjectDetection(std::string model_json_path, std::string model_path, std::string data_json_path){
  vector<CaseObject> cases = readCases(data_json_path);
	JSONObject *model_json = new JSONObject();
	std::vector <json_token_t*> model_tokens = model_json->load(model_json_path);
  float learning_rate = 0.001;
  float momentum = 0.6;
  float decay = 0.0005;
	vector<LayerObject*> layers = loadModel(model_json, model_tokens, cases[0], learning_rate, decay, momentum);
}

void testObjectDetection(string model_config_path, string model_path, string classes_path, string data_config_path, float threshold){
  printf("test!\n");
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

  ArgumentProcessor *argument_processor = new ArgumentProcessor();
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
