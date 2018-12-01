#include <iostream>
#include <vector>
using namespace std;

#include "lightnet.hpp"
#include "ArgumentProcessor.hpp"
#include "ImageProcessor.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"

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
    if(utils->fileExists(label_path)){
      cout << image_path << endl;
      cout << label_path << endl;
      image_paths.push_back(image_path);
    }else{
      cout << "Not exists:" + label_path << endl;
    }
  }
  delete utils;
}

void trainObjectDetection(std::string model_config_path, std::string model_path, std::string data_config_path){
  Utils *utils = new Utils();
  // load data config
  JSONObject *data_json = new JSONObject();
  std::vector <JSONObject::json_token_t*> data_tokens = data_json->load(data_config_path);
  printf("- train_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "train_dir")).c_str());   // data_tokens[0] := json root
  printf("- checkpoint_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "checkpoint_dir")).c_str());
  printf("- test_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "test_dir")).c_str());
  printf("- test_results_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "test_results_dir")).c_str());

  // load network model config
  std::vector<string> train_image_paths;
  std::vector<string> train_label_paths;
  getDataList(data_json->getChildValueForToken(data_tokens[0], "train_dir"), train_image_paths, train_label_paths);
  ImageProcessor *image_processor = new ImageProcessor();
  for(int i=0; i<train_image_paths.size(); i++){
    ImageProcessor::image_st image = image_processor->readImageFile(train_image_paths[i], 416, 416, 3);
    printf("Read image: %s %d %d\n", train_image_paths[i].c_str(), image.width, image.height);
    // image_processor->writeImageFilePNG(utils->stringReplace(train_image_paths[i], "jpg", "png"), image);
  }

  NeuralNetwork *nn = new NeuralNetwork();
  nn->loadModel(model_config_path, model_path);
  delete nn;
  delete utils;
}

void testObjectDetection(string model_config_path, string model_path, string classes_path, string data_config_path, float threshold){
  printf("test!\n");
}

void doObjectDetection(int argc, char **argv)
{
  if(argc < 6){
    fprintf(stderr, "usage: %s %s <train|test> <model_config_path> <model_path> <data_config_path> [other options...]\n", argv[0], argv[1]);
    return;
  }
  string model_config_path = argv[3];
  string model_path = argv[4];
  string data_config_path = argv[5];

  ArgumentProcessor *argumentProcessor = new ArgumentProcessor();
  string classes_path = argumentProcessor->get_chars_parameter(argc, argv, "-classes_path", "classes.txt");
  float threshold = argumentProcessor->get_float_parameter(argc, argv, "-threshold", .3);
  delete argumentProcessor;

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
