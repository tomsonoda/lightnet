using namespace std;
class ImageProcessor
{
private:
public:
  typedef struct {
    int width;
    int height;
    int channels;
    float *data;
  } image_st;

  ImageProcessor();
  ~ImageProcessor();
  image_st readImageFile(std::string image_filename, int target_width, int target_height, int target_channels);
  void writeImageFilePNG(std::string image_filename, ImageProcessor::image_st);
};
