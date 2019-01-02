#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

#pragma pack(push, 1)
typedef struct {
  int width;
  int height;
  int channels;
  float *data;
} image_st;
#pragma pack(pop)

struct ImageProcessor
{
  image_st readImageFile(std::string image_filename, int target_width, int target_height, int target_channels)
  {
    int width, height, channels;
    unsigned char *data = stbi_load(image_filename.c_str(), &width, &height, &channels, target_channels);
    if (!data) {
      fprintf(stderr, "Read failed: %s : %s\n", image_filename.c_str(), stbi_failure_reason());
      exit(0);
    }
    if(target_channels){
      channels = target_channels;
    }
    // load
    image_st image;
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.data = (float *)calloc(width*height*channels, sizeof(float));
    for(int c=0; c<channels; c++){
      for(int h=0; h<height; h++){
        for(int w=0; w<width; w++){
          int org_i = (channels*width*h) + (channels*w) + c;
          int new_i = (width*height*c) + (width*h) + w;
          image.data[new_i] = (float)data[org_i]/255.0;
        }
      }
    }
    free(data);
    return image;
  }

  void writeImageFilePNG(std::string image_filename, image_st image)
  {
    unsigned char *data = (unsigned char *)calloc(image.width * image.height * image.channels, sizeof(char));
    for(int c=0; c<image.channels; c++){
      for(int i=0; i<image.width*image.height; ++i){
        data[i*image.channels+c] = (unsigned char) (255*image.data[i + c*image.width*image.height]);
      }
    }
    int success = stbi_write_png(image_filename.c_str(), image.width, image.height, image.channels, data, image.width*image.channels);
    free(data);
    if(!success){
      fprintf(stderr, "Write failed : %s\n", image_filename.c_str());
    }
  }
};
