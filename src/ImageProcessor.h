#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/*
 note: This uses stb image library that has non-inline methods. We recommend to include this in cpp/c file. (not in header files.)
*/

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
  void free_image(image_st m)
  {
    if(m.data){
      free(m.data);
    }
  }

  static void add_pixel(image_st m, int x, int y, int c, float val)
  {
    assert(x < m.width && y < m.height && c < m.channels);
    m.data[c*m.height * m.width + y*m.width + x] += val;
  }

  static float get_pixel(image_st m, int x, int y, int c)
  {
    assert(x < m.width && y < m.height && c < m.channels);
    return m.data[c * m.height * m.width + y * m.width + x];
  }

  static void set_pixel(image_st m, int x, int y, int c, float val)
  {
      if (x < 0 || y < 0 || c < 0 || x >= m.width || y >= m.height || c >= m.channels){
        return;
      }
      assert(x < m.width && y < m.height && c < m.channels);
      m.data[c*m.height * m.width + y * m.width + x] = val;
  }

  image_st make_empty_image(int w, int h, int c)
  {
    image_st out;
    out.data = 0;
    out.height = h;
    out.width = w;
    out.channels = c;
    return out;
  }

  image_st make_image(int w, int h, int c)
  {
    image_st out = make_empty_image(w,h,c);
    out.data = (float *)calloc(h*w*c, sizeof(float));
    return out;
  }

  image_st resize_image(image_st im, int w, int h)
  {
    image_st resized = make_image(w, h, im.channels);
    image_st part = make_image(w, im.height, im.channels);
    int r, c, k;
    float w_scale = (float)(im.width - 1) / (w - 1);
    float h_scale = (float)(im.height - 1) / (h - 1);
    for(k = 0; k < im.channels; ++k){
        for(r = 0; r < im.height; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.width == 1){
                    val = get_pixel(im, im.width-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.channels; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.height == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
  }

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

    if((target_height && target_width) && (target_height != image.height || target_width != image.width)){
      image_st resized = resize_image(image, target_width, target_height);
      free_image(image);
      image = resized;
    }

    free(data);
    return image;
  }

  void writeImageFilePNG(std::string image_filename, image_st image)
  {
    unsigned char *data = (unsigned char *)calloc(image.width * image.height * image.channels, sizeof(char));
    for(int c=0; c<image.channels; ++c){
      for(int i=0; i<image.width*image.height; ++i){
        data[i*image.channels+c] = (unsigned char) (255*image.data[i + c*image.width*image.height]);
      }
    }
    printf("Image width=%d, height=%d, channels=%d\n", image.width, image.height, image.channels);
    int success = stbi_write_png(image_filename.c_str(), image.width, image.height, image.channels, data, image.width*image.channels);
    free(data);
    if(!success){
      fprintf(stderr, "Write failed : %s\n", image_filename.c_str());
    }
  }
};
