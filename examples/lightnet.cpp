#include <iostream>

using namespace std;

extern void classification(int argc, char **argv);  // classification.cpp
extern void objectDetection(int argc, char **argv); // object_detection.cpp

int main(int argc, char **argv)
{
  if(argc < 4){
    fprintf(stderr, "usage: %s <classification | object_detection> <data_json_path> <model_json_path [check_point_path]\n", argv[0]);
    return 0;
  }

  if(strcmp(argv[1], "classification")==0){

    classification(argc, argv);

  }else if (strcmp(argv[1], "object_detection")==0){

    objectDetection(argc, argv);

  }
}
