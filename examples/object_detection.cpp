#include <iostream>
#include <fstream>
#include <vector>
#include "lightnet.hpp"
#include "ImageProcessor.hpp"

using namespace std;

void getDataList(std::string dir_string, std::vector<string> &image_paths, std::vector<string> &label_paths)
{
  Utils *utils = new Utils();
  if (utils->stringEndsWith(dir_string, "/")==0){
    dir_string = dir_string + "/";
  }
  std::string image_dir_string = dir_string + "images/";
  std::string label_dir_string = dir_string + "labels/";
  std::string image_ext = "jpg";
  std::string label_ext = "txt";

  vector<string> files = vector<string>();
  utils->listDir(image_dir_string,files,image_ext);
  for (int j=0; j<files.size(); j++){
    std::string image_path = files[j];
    std::string label_path = files[j];
    image_path = image_dir_string + image_path;
    label_path = label_dir_string + utils->stringReplace(label_path, image_ext, label_ext);

    std::ifstream ifs(label_path);
    if(ifs.is_open()){ // if exists
      cout << image_path << endl;
      cout << label_path << endl;
      image_paths.push_back(image_path);
    }else{
      cout << "Not exists:" + label_path << endl;
    }
  }
  delete utils;
}

float train( vector<layer_t*>& layers, tensor_t<float>& data, tensor_t<float>& expected)
{
  for (int i=0; i<layers.size(); i++){
    if (i== 0){
      activate(layers[i], data);
    }else{
      activate(layers[i], layers[i-1]->out);
    }
  }
  tensor_t<float> grads = layers.back()->out - expected;

  for (int i=layers.size()-1; i>=0; i--){
    if (i==layers.size()-1){
      calc_grads(layers[i], grads);
    }else{
      calc_grads(layers[i], layers[i+1]->grads_in);
    }
  }

  for(int i=0; i<layers.size(); i++){
    fix_weights(layers[i]);
  }

  float err = 0;
  for( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++ ){
    float f = expected.data[i];
    if ( f > 0.5 ){
      err += abs(grads.data[i]);
    }
  }
  return err * 100;
}
/*
vector<case_t> readCases(std::string data_config_path)
{
  JSONObject *data_json = new JSONObject();
  std::vector <json_token_t*> data_tokens = data_json->load(data_config_path);
  printf("- train_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "train_dir")).c_str());   // data_tokens[0] := json root
  printf("- checkpoint_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "checkpoint_dir")).c_str());
  printf("- test_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "test_dir")).c_str());
  printf("- test_results_dir : %s\n", (data_json->getChildValueForToken(data_tokens[0], "test_results_dir")).c_str());

  std::vector<string> train_image_paths;
  std::vector<string> train_label_paths;
  getDataList(data_json->getChildValueForToken(data_tokens[0], "train_dir"), train_image_paths, train_label_paths);
  ImageProcessor *image_processor = new ImageProcessor();
  for(int i=0; i<train_image_paths.size(); i++){
    image_st image = image_processor->readImageFile(train_image_paths[i], 416, 416, 3);
    printf("Read image: %s %d %d\n", train_image_paths[i].c_str(), image.width, image.height);
    // image_processor->writeImageFilePNG(utils->stringReplace(train_image_paths[i], "jpg", "png"), image);
  }
  vector<case_t> cases;
  return cases;
}
*/

void trainObjectDetection(std::string model_config_path, std::string model_path, std::string data_config_path){
  // Utils *utils = new Utils();
  // vector<case_t> cases = readCases(data_config_path);
  // lightnet_t *lightnet_object = new lightnet_t();
  // vector<layer_t*> layers = lightnet_object->loadModel(model_config_path, cases);

  /*
	float amse = 0;
	int ic = 0;

	for(long ep=0; ep <100000;){
		for (case_t& t : cases){
			float xerr = train( layers, t.data, t.out );
			amse += xerr;

			ep++;
			ic++;

			if ( ep % 1000 == 0 ){
        cout << "case " << ep << " err=" << amse/ic << endl;
      }
		}
	}
	// end:

	while(true){
		uint8_t * data = read_file( "test.ppm" );
		if ( data ){
			uint8_t * usable = data;
			while ( *(uint32_t*)usable != 0x0A353532 ){
        usable++;
      }

#pragma pack(push, 1)
			struct RGB
			{
				uint8_t r, g, b;
			};
#pragma pack(pop)

			RGB * rgb = (RGB*)usable;
			tensor_t<float> image(28, 28, 1);
			for ( int i = 0; i < 28; i++ ){
				for ( int j = 0; j < 28; j++ ){
					RGB rgb_ij = rgb[i * 28 + j];
					image( j, i, 0 ) = (((float)rgb_ij.r
							     + rgb_ij.g
							     + rgb_ij.b)
							    / (3.0f*255.f));
				}
			}

			forward( layers, image );
			tensor_t<float>& out = layers.back()->out;
			for ( int i = 0; i < 10; i++ ){
				printf( "[%i] %f\n", i, out( i, 0, 0 )*100.0f );
			}
			delete[] data;
		}

		struct timespec wait;
		wait.tv_sec = 1;
		wait.tv_nsec = 0;
		nanosleep(&wait, nullptr);
	}
  */
  // delete utils;
}

void testObjectDetection(string model_config_path, string model_path, string classes_path, string data_config_path, float threshold){
  printf("test!\n");
}

void objectDetection(int argc, char **argv)
{
  if(argc < 6){
    fprintf(stderr, "usage: %s %s <train|test> <data_config_path> <model_config_path> <model_path> [other options...]\n", argv[0], argv[1]);
    return;
  }
	string data_config_path = argv[3];
  string model_config_path = argv[4];
  string model_path = argv[5];

  ArgumentProcessor *argument_processor = new ArgumentProcessor();
  string classes_path = argument_processor->get_chars_parameter(argc, argv, "-classes_path", "classes.txt");
  float threshold = argument_processor->get_float_parameter(argc, argv, "-threshold", .3);
  delete argument_processor;

  if(strcmp(argv[2], "train")==0){
    if(argc>5){
      trainObjectDetection(model_config_path, model_path, data_config_path);
    }else{
      fprintf(stderr, "usage: %s %s train <model_config_path> <model_path> <data_config_path> [other options]\n", argv[0], argv[1]);
      return;
    }
  }else if(strcmp(argv[2], "test")==0){
    if(argc>5){
      testObjectDetection(model_config_path, model_path, classes_path, data_config_path, threshold);
    }else{
      fprintf(stderr, "usage: %s %s test <model_config_path> <model_path> <data_config_path> [other options]\n", argv[0], argv[1]);
      return;
    }
  }
}
