#pragma once
#include "ArgumentProcessor.hpp"
#include "JSONObject.hpp"
#include "Utils.hpp"
#include "cnn.h"

struct lightnet_t
{
  vector<layer_t*> loadModel(std::string model_config_path, vector<case_t> cases)
  {
    vector<layer_t*> layers;
    if(cases.size()==0){
      return layers;
    }

  	std::vector<json_token_t*> json_layers;
  	JSONObject *model_json = new JSONObject();
  	std::vector <json_token_t*> model_tokens = model_json->load(model_config_path);
  	json_layers = model_json->getArrayForToken(model_tokens[0], "layers");

  	for(int i=0; i<json_layers.size(); i++){
  		std::string type = model_json->getChildValueForToken(json_layers[i], "type");
      if(type=="convolutional"){
        // std::string batch_normalize = model_json->getChildValueForToken(json_layers[i], "batch_normalize");
        uint16_t stride = std::stoi( model_json->getChildValueForToken(json_layers[i], "stride") );
        uint16_t size = std::stoi( model_json->getChildValueForToken(json_layers[i], "size") );
        uint16_t filters = std::stoi( model_json->getChildValueForToken(json_layers[i], "filters") );
        uint16_t padding = std::stoi( model_json->getChildValueForToken(json_layers[i], "pad") );
        tdsize input_size;
        if(i==0){
          input_size = cases[0].data.size;
        }else{
          input_size = layers[layers.size()-1]->out.size;
        }
        printf("%d: convolutional stride=%d  extend_filter=%d filters=%d pad=%d\n", i, stride, size, filters, padding);
        conv_layer_t * layer = new conv_layer_t( stride, size, filters, padding, input_size);		// 28 * 28 * 1 -> 24 * 24 * 8
        layers.push_back( (layer_t*)layer );

      }else if(type=="relu"){
        printf("%d: relu \n", i);
        relu_layer_t * layer = new relu_layer_t( layers[layers.size()-1]->out.size );
        layers.push_back( (layer_t*)layer );

      }else if(type=="maxpool"){
        uint16_t stride = std::stoi( model_json->getChildValueForToken(json_layers[i], "stride") );
        uint16_t size = std::stoi( model_json->getChildValueForToken(json_layers[i], "size") );
        tdsize input_size = layers[layers.size()-1]->out.size;
        printf("%d: maxpool stride=%d  extend_filter=%d \n", i, stride, size );
        pool_layer_t * layer = new pool_layer_t( stride, size, input_size );				// 24 * 24 * 8 -> 12 * 12 * 8
        layers.push_back( (layer_t*)layer );

      }else if(type=="fully_connected"){
        tdsize in_size = layers[layers.size()-1]->out.size;
        int out_size = cases[0].out.size.x * cases[0].out.size.y * cases[0].out.size.z;
        printf("%d: fully_connected -> %d\n",i, out_size);
        fc_layer_t * layer = new fc_layer_t(in_size, out_size);					// 4 * 4 * 16 -> 10
        layers.push_back( (layer_t*)layer );

      }else if (type=="route"){
  			std::vector <json_token_t*> ref_layers = model_json->getArrayForToken(json_layers[i], "layers");
  			for(int j=0; j<ref_layers.size(); j++){
  				std::string value = model_json->getValueForToken(ref_layers[j]);
  				printf("%s ", value.c_str());
  			}
  			printf("\n");
  		}
  	}

    return layers;
  }
};
