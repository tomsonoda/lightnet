#include <iostream>

using namespace std;
extern void objectDetection(int argc, char **argv); // object_detection.cpp
extern void mnist(int argc, char **argv);           // mnist.cpp

int main(int argc, char **argv)
{
  if(argc < 4){
    fprintf(stderr, "usage: %s object_detection train <data_config_path> <model_config_path> <model_path> [other options]\n", argv[0]);
    fprintf(stderr, "       %s mnist <data_config_path> <model_config_path> <model_path>\n", argv[0]);
    return 0;
  }

  printf("lightnet begins.\n");
  if (strcmp(argv[1], "object_detection")==0){
    objectDetection(argc, argv);
  }else if(strcmp(argv[1], "mnist")==0){
    mnist(argc, argv);
  }
  printf("lightnet ends.\n");
}
