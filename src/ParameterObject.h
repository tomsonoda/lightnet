#pragma once
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm> // std::equal

#pragma pack(push, 1)
struct ParameterObject
{
  int save_latest_span;
  int save_span;

  unsigned batch_size;
  float learning_rate;
  float momentum;
  float weights_decay;
  string optimizer;
  int train_output_span;
  int threads;
  int max_classes;
  int max_bounding_boxes;

  ParameterObject()
  {
    save_latest_span = 100;
    save_span = 1000;
    batch_size = 1;
    learning_rate = 0.01;
    momentum = 0.6;
    weights_decay = 0.01;
    optimizer = "mse";
    train_output_span = 1000;
    threads = 1;
    max_classes = 1;
    max_bounding_boxes = 30;
  }

  ~ParameterObject()
  {
  }

  void printParameters()
  {
    printf("batch_size         : %d\n", batch_size);
  	printf("threads            : %d\n", threads);
  	printf("learning_rate      : %f\n", learning_rate);
  	printf("momentum           : %f\n", momentum);
  	printf("weights_decay      : %f\n", weights_decay);
  	printf("optimizer          : %s\n\n", optimizer.c_str());
    printf("max_classes        : %d\n", max_classes);
  	printf("max_bounding_boxes : %d\n\n", max_bounding_boxes);
  }

  float getFloatParameter(int argc, char **argv, string name, float default_value)
  {
    int i;
    float value = default_value;
    for(i=0; i<argc-1; ++i){
        if(equal(string(argv[i]).begin(), string(argv[i]).end(), name.begin())){
            value = atof(argv[i+1]);
            break;
        }
    }
    return value;
  }

  string getCharsParameter(int argc, char **argv, string name, string default_value)
  {
    int i;
    string value = default_value;
    for(i=0; i<argc-1; ++i){
        if(equal(string(argv[i]).begin(), string(argv[i]).end(), name.begin())){
            value = string(argv[i+1]);
            break;
        }
    }
    return value;
  }
} ;
#pragma pack(pop)
