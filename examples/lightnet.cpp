#include <iostream>

using namespace std;

extern void classification(int argc, char **argv);           // mnist.cpp
extern void objectDetection(int argc, char **argv); // object_detection.cpp

int main(int argc, char **argv)
{
  if(argc < 4){
    fprintf(stderr, "usage: %s classification <data_json_path> <model_json_path [checkpoints_dir]\n", argv[0]);
    fprintf(stderr, "       %s object_detection <train|test> <data_json_path> <model_json_path> <model_path> [other options]\n", argv[0]);
    return 0;
  }

  if(strcmp(argv[1], "classification")==0){

    classification(argc, argv);

  }else if (strcmp(argv[1], "object_detection")==0){

    objectDetection(argc, argv);

  }

  printf("lightnet ends.\n");
}
