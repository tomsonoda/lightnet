#include "parameters.h"
#include <string.h>
#include <stdlib.h>

float get_float_parameter(int argc, char **argv, char *name, float default_value){
  int i;
  float value = default_value;
  for(i=0; i<argc-1; i++){
      if(strcmp(argv[i], name)==0){
          value = atof(argv[i+1]);
          break;
      }
  }
  return value;
}

char *get_chars_parameter(int argc, char **argv, char *name, char *default_value)
{
  int i;
  char *value = default_value;
  for(i=0; i<argc-1; i++){
      if(strcmp(argv[i], name)==0){
          value = argv[i+1];
          break;
      }
  }
  return value;
}
