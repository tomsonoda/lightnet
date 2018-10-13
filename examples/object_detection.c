#include "lightnet.h"
#include "parameters.h"

void train_object_detection(char *model_config_path, char *model_path, char *data_config_path){
  printf("train!\n");
}

void test_object_detection(char *model_config_path, char *model_path, char *classes_path, char *data_config_path, float threshold){
  printf("test!\n");
}

void do_object_detection(int argc, char **argv)
{
    if(argc < 6){
        fprintf(stderr, "usage: %s %s <train|test> <model_config_path> <model_path> <data_config_path> [other options...]\n", argv[0], argv[1]);
        return;
    }
    char *model_config_path = argv[3];
    char *model_path = argv[4];
    char *data_config_path = argv[5];
    char *classes_path = get_chars_parameter(argc, argv, "-classes_path", "classes.txt");
    float threshold = get_float_parameter(argc, argv, "-threshold", .3);

    if(strcmp(argv[2], "train")==0){
      if(argc>5){
        train_object_detection(model_config_path, model_path, data_config_path);
      }else{
        fprintf(stderr, "usage: %s %s train <model_config_path> <model_path> <data_config_path> [other options]\n", argv[0], argv[1]);
        return;
      }
    }
    else if(strcmp(argv[2], "test")==0){
      if(argc>5){
        test_object_detection(model_config_path, model_path, classes_path, data_config_path, threshold);
      }else{
        fprintf(stderr, "usage: %s %s test <model_config_path> <model_path> <data_config_path> [other options]\n", argv[0], argv[1]);
        return;
      }
    }
}
