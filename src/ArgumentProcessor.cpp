#include "ArgumentProcessor.hpp"
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm> // std::equal

using namespace std;

ArgumentProcessor::ArgumentProcessor()
{
}

ArgumentProcessor::~ArgumentProcessor()
{
}

float ArgumentProcessor::get_float_parameter(int argc, char **argv, string name, float default_value){
  int i;
  float value = default_value;
  for(i=0; i<argc-1; i++){
      if(equal(string(argv[i]).begin(), string(argv[i]).end(), name.begin())){
          value = atof(argv[i+1]);
          break;
      }
  }
  return value;
}

string ArgumentProcessor::get_chars_parameter(int argc, char **argv, string name, string default_value)
{
  int i;
  string value = default_value;
  for(i=0; i<argc-1; i++){
      if(equal(string(argv[i]).begin(), string(argv[i]).end(), name.begin())){
          value = string(argv[i+1]);
          break;
      }
  }
  return value;
}
