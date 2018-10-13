#include "lightnet.h"

extern void do_object_detection(int argc, char **argv);

int main(int argc, char **argv)
{
  if(argc < 3){
    fprintf(stderr, "usage: %s object_detection train <model_config_path> <model_path> <data_config_path> [other options]\n", argv[0]);
    return 0;
  }

  printf("lightnet begins.\n");
  if (strcmp(argv[1], "object_detection")==0){
      do_object_detection(argc, argv);
  }
  printf("lightnet ends.\n");
  // error("lightnet terminated!");
}
