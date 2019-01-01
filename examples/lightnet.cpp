#include <iostream>

using namespace std;

extern void mnist(int argc, char **argv);           // mnist.cpp
extern void cifar(int argc, char **argv);           // mnist.cpp
extern void objectDetection(int argc, char **argv); // object_detection.cpp

int main(int argc, char **argv)
{
  if(argc < 4){
    fprintf(stderr, "usage: %s mnist <data_json_path> <model_json_path>\n", argv[0]);
    fprintf(stderr, "       %s cifar10 <data_json_path> <model_json_path>\n", argv[0]);
    fprintf(stderr, "       %s object_detection <train|test> <data_json_path> <model_json_path> <model_path> [other options]\n", argv[0]);
    return 0;
  }

  if(strcmp(argv[1], "mnist")==0){
    mnist(argc, argv);
  }
  if(strcmp(argv[1], "cifar")==0){
    cifar(argc, argv);
  }
  if (strcmp(argv[1], "object_detection")==0){
    objectDetection(argc, argv);
  }
  printf("lightnet ends.\n");
}
