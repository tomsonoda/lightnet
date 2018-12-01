#include <stdio.h>
#include <string>
#include "JSONObject.hpp"

using namespace std;

class NeuralNetwork
{
private:

public:
  int n;
  int *t;
  std::vector <JSONObject::json_token_t*> layers;
  int batch;
  size_t *seen;
  float epoch;
  int subdivisions;
  float *output;
  float *cost;
  // learning_rate_policy policy;

  float learning_rate;
  float momentum;
  float decay;
  float gamma;
  float scale;
  float power;
  int time_steps;
  int step;
  int max_batches;
  float *scales;
  int   *steps;
  int num_steps;
  int burn_in;

  int adam;
  float B1;
  float B2;
  float eps;

  int inputs;
  int outputs;
  int truths;
  int notruth;
  int h, w, c;
  int max_crop;
  int min_crop;
  float max_ratio;
  float min_ratio;
  int center;
  float angle;
  float aspect;
  float exposure;
  float saturation;
  float hue;
  int random;
  // tree *hierarchy;

  int gpu_index;
  float *input;
  float *truth;
  float *delta;
  float *workspace;
  int train;
  int index;

  NeuralNetwork();
  ~NeuralNetwork();
  void loadModel(const std::string model_config_path, const std::string model_config);
};
