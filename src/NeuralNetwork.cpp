#include <iostream>
#include "NeuralNetwork.hpp"
#include "Utils.hpp" // output log

using namespace std;

NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::loadModel(const std::string model_config_path, const std::string model_config)
{
  Utils *utils = new Utils();
  utils->outputLog(__FILE__, __FUNCTION__, __LINE__, "");
  JSONObject *model_json = new JSONObject();
  std::vector <JSONObject::json_token_t*> model_tokens = model_json->load(model_config_path);
  this->layers = model_json->getArrayForToken(model_tokens[0], "layers");
  for(int i=0; i<layers.size(); i++){
    std::string type = model_json->getChildValueForToken(layers[i], "type");
    std::string batch_normalize = model_json->getChildValueForToken(layers[i], "batch_normalize");
    std::string filters = model_json->getChildValueForToken(layers[i], "filters");
    std::string size = model_json->getChildValueForToken(layers[i], "size");
    std::string stride = model_json->getChildValueForToken(layers[i], "stride");
    std::string pad = model_json->getChildValueForToken(layers[i], "pad");
    std::string activation = model_json->getChildValueForToken(layers[i], "activation");
    printf("%d: [%s] %s %s %s %s %s %s\n", i, type.c_str(), batch_normalize.c_str(), filters.c_str(), size.c_str(), stride.c_str(), pad.c_str(), activation.c_str());
    if (type=="route"){
      std::vector <JSONObject::json_token_t*> ref_layers = model_json->getArrayForToken(layers[i], "layers");
      for(int j=0; j<ref_layers.size(); j++){
        std::string value = model_json->getValueForToken(ref_layers[j]);
        printf("%s ", value.c_str());
      }
      printf("\n");
    }
  }
  delete utils;
}
