#include <iostream>

#include "BoundingBoxObject.h"
#include "CaseObject.h"
#include "ImageProcessor.h"
#include "JSONObject.h"
#include "Utils.h"

#define OBJECT_DETECTION_IMAGE_WIDTH    (416)
#define OBJECT_DETECTION_IMAGE_HEIGHT   (416)
#define OBJECT_DETECTION_MAX_BOUNDBOXES (30)

using namespace std;

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float boxIntersection(boundingbox_t a, boundingbox_t b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float boxUnion(boundingbox_t a, boundingbox_t b)
{
    float i = boxIntersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float boxIOU(boundingbox_t a, boundingbox_t b)
{
    return boxIntersection(a, b)/boxUnion(a, b);
}

float boxTensorIOU(TensorObject<float> &t_a, TensorObject<float> &t_b, JSONObject *model_json, vector<json_token_t*> model_tokens )
{
  int classes = 1;
  int max_bounding_boxes = OBJECT_DETECTION_MAX_BOUNDBOXES;

  json_token_t* nueral_network = model_json->getChildForToken(model_tokens[0], "net");
  string tmp = model_json->getChildValueForToken(nueral_network, "max_classes");
  if(tmp.length()>0){
    classes = std::stoi( tmp );
  }
  tmp = model_json->getChildValueForToken(nueral_network, "max_bounding_boxes");
  if(tmp.length()>0){
    max_bounding_boxes = std::stoi( tmp );
  }

  float best_iou = 0.0;

  for( int j=0; j<max_bounding_boxes; ++j){
    int index = j * (4 + classes);
    boundingbox_t a, b;
    a.x = t_a( 0, index, 0, 0 );
    a.y = t_a( 0, index+1, 0, 0 );
    a.w = t_a( 0, index+2, 0, 0 );
    a.h = t_a( 0, index+3, 0, 0 );

    b.x = t_b( 0, index, 0, 0 );
    b.y = t_b( 0, index+1, 0, 0 );
    b.w = t_b( 0, index+2, 0, 0 );
    b.h = t_b( 0, index+3, 0, 0 );
    float iou = boxIOU(a, b);
    if( iou > best_iou ){
      best_iou = iou;
    }
  }

  return best_iou;
}

uint32_t reverseUint32(uint32_t a)
{
	return ((((a >> 24) & 0xff) << 0) |
		(((a >> 16) & 0xff) << 8) |
		(((a >> 8) & 0xff) << 16) |
		(((a >> 0) & 0xff) << 24));
}

uint8_t* readFileToBuffer( const char* szFile ){
	ifstream file( szFile, ios::binary | ios::ate );
	streamsize size = file.tellg();
	file.seekg( 0, ios::beg );

	if ( size == -1 ){
		return nullptr;
	}

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size );
	return buffer;
}

vector<CaseObject> readCasesMNIST( JSONObject *data_json, vector <json_token_t*> data_tokens, string mode )
{
	Utils *utils = new Utils();
	vector<CaseObject> cases;

	uint8_t* images;
	uint8_t* labels;

	if(mode=="train"){
		string train_dir = data_json->getChildValueForToken(data_tokens[0], "train_dir");   // data_tokens[0] := json root
		if (utils->stringEndsWith(train_dir, "/")==0){
			train_dir = train_dir + "/";
		}
		images = readFileToBuffer( (train_dir + "train-images-idx3-ubyte").c_str() );
		labels = readFileToBuffer( (train_dir + "train-labels-idx1-ubyte").c_str() );
	}else{
		string test_dir = data_json->getChildValueForToken(data_tokens[0], "test_dir");   // data_tokens[0] := json root
		if (utils->stringEndsWith(test_dir, "/")==0){
			test_dir = test_dir + "/";
		}
		images = readFileToBuffer( (test_dir + "t10k-images-idx3-ubyte").c_str() );
		labels = readFileToBuffer( (test_dir + "t10k-labels-idx1-ubyte").c_str() );
	}

	uint32_t case_count = reverseUint32( *(uint32_t*)(images + 4) );

	for (uint32_t i=0; i<case_count; ++i){
		CaseObject c {TensorObject<float>( 1, 28, 28, 1 ), TensorObject<float>( 1, 10, 1, 1 )};
		uint8_t* img = images + 16 + i * (28 * 28);
		uint8_t* label = labels + 8 + i;
		for ( int x = 0; x < 28; x++ ){
			for ( int y = 0; y < 28; y++ ){
				c.data( 0, x, y, 0 ) = img[x + y * 28] / 255.f;
			}
		}
		for ( int b = 0; b < 10; ++b ){
			c.out( 0, b, 0, 0 ) = *label == b ? 1.0f : 0.0f;
		}
		cases.push_back( c );
	}

	delete[] images;
	delete[] labels;
	delete utils;
	return cases;
}

vector<CaseObject> readCasesCifar10( JSONObject *data_json, vector<json_token_t*> data_tokens, string mode )
{
	Utils *utils = new Utils();
	vector<CaseObject> cases;

	uint8_t* buffer;
	vector<uint8_t> v;
	streamsize file_size;
	if(mode=="train"){
		string train_dir = data_json->getChildValueForToken(data_tokens[0], "train_dir");   // data_tokens[0] := json root
		if (utils->stringEndsWith(train_dir, "/")==0){
			train_dir = train_dir + "/";
		}

		const char *file_name = (train_dir + "data_batch_1.bin").c_str();
		ifstream file1( file_name, ios::binary | ios::ate );
		long file_size1 = file1.tellg();
		uint8_t* buffer1 = readFileToBuffer( file_name );

		file_name = (train_dir + "data_batch_2.bin").c_str();
		ifstream file2( file_name, ios::binary | ios::ate );
		long file_size2 = file2.tellg();
		uint8_t* buffer2 = readFileToBuffer( file_name );

		file_name = (train_dir + "data_batch_3.bin").c_str();
		ifstream file3( file_name, ios::binary | ios::ate );
		long file_size3 = file3.tellg();
		uint8_t* buffer3 = readFileToBuffer( file_name );

		file_name = (train_dir + "data_batch_4.bin").c_str();
		ifstream file4( file_name, ios::binary | ios::ate );
		long file_size4 = file4.tellg();
		uint8_t* buffer4 = readFileToBuffer( file_name );

		file_name = (train_dir + "data_batch_5.bin").c_str();
		ifstream file5( file_name, ios::binary | ios::ate );
		long file_size5 = file5.tellg();
		uint8_t* buffer5 = readFileToBuffer( file_name );

		int pointer = 0;
		file_size = file_size1 + file_size2 + file_size3 + file_size4 + file_size5;
		buffer = new uint8_t[file_size];
		memcpy(buffer, buffer1, file_size1);
		pointer += file_size1;
		memcpy(buffer+pointer, buffer2, file_size2);
		pointer += file_size2;
		memcpy(buffer+pointer, buffer3, file_size3);
		pointer += file_size3;
		memcpy(buffer+pointer, buffer4, file_size4);
		pointer += file_size4;
		memcpy(buffer+pointer, buffer5, file_size5);

		delete[] buffer1;
		delete[] buffer2;
		delete[] buffer3;
		delete[] buffer4;
		delete[] buffer5;

	}else{
		string test_dir = data_json->getChildValueForToken(data_tokens[0], "test_dir");   // data_tokens[0] := json root
		const char *file_name = (test_dir + "test_batch.bin").c_str();
		ifstream file( file_name, ios::binary | ios::ate );
		file_size = file.tellg();
		buffer = readFileToBuffer( file_name );
	}
	uint32_t case_count = file_size / (32 * 32 * 3 + 1);
	for (uint32_t i=0; i<case_count; ++i){
		CaseObject c {TensorObject<float>( 1, 32, 32, 3 ), TensorObject<float>( 1, 10, 1, 1 )};
		uint8_t* img = buffer + i * (32 * 32 * 3 + 1) + 1;
		uint8_t* label = buffer + i * (32 * 32 * 3 + 1);
		for(int z = 0; z < 3; ++z ){
			for ( int y = 0; y < 32; y++ ){
				for ( int x = 0; x < 32; x++ ){
					c.data( 0, x, y, z ) = img[x + (y * 32) + (32 * 32)*z] / 255.f;
				}
			}
		}
		for ( int b = 0; b < 10; ++b ){
			c.out( 0, b, 0, 0 ) = *label == b ? 1.0f : 0.0f;
		}
		cases.push_back( c );
	}
	delete[] buffer;
	delete utils;
	return cases;
}

vector<boundingbox_t> readLabelBoxes(string filename)
{
		FILE *file = fopen(filename.c_str(), "r");
		if(!file){
			printf("%s not found.\n",filename.c_str());
			exit(0);
		}
		float x, y, h, w;
		int box_id;
		vector<boundingbox_t> boxes = vector<boundingbox_t>();
		while(fscanf(file, "%d %f %f %f %f", &box_id, &x, &y, &w, &h) == 5){
			boundingbox_t box;
			box.id = box_id;
			box.x  = x;
			box.y  = y;
			box.h  = h;
			box.w  = w;
			box.left   = x - w/2;
			box.right  = x + w/2;
			box.top    = y - h/2;
			box.bottom = y + h/2;
			boxes.push_back(box);
		}
		fclose(file);
		return boxes;
}

vector<CasePaths> listImageLabelCasePaths( JSONObject *data_json, vector<json_token_t*> data_tokens, JSONObject *model_json, vector<json_token_t*> model_tokens, string mode )
{
  Utils *utils = new Utils();
  vector<CasePaths> case_paths_list;

	string dir_string;
	if(mode=="train"){
		dir_string = data_json->getChildValueForToken(data_tokens[0], "train_dir");   // data_tokens[0] := json root
	}else{ // test cases
		dir_string = data_json->getChildValueForToken(data_tokens[0], "test_dir");   // data_tokens[0] := json root
	}
	if (utils->stringEndsWith(dir_string, "/")==0){
		dir_string = dir_string + "/";
	}
	std::string image_dir_string = dir_string + "images/";
	std::string label_dir_string = dir_string + "labels/";
	std::string image_ext = "jpg";
	std::string label_ext = "txt";

	vector<string> files = vector<string>();
	vector<string> image_paths;
	vector<string> label_paths;

	utils->listDir(image_dir_string,files,image_ext);

	for (int j=0; j<files.size(); ++j){
		std::string image_path = files[j];
		std::string label_path = files[j];
		image_path = image_dir_string + image_path;
		label_path = label_dir_string + utils->stringReplace(label_path, image_ext, label_ext);

		std::ifstream ifs(label_path);
		if(ifs.is_open()){ // if exists
			// cout << image_path << endl;
			// cout << label_path << endl;
      CasePaths case_paths { image_path, label_path };
      case_paths_list.push_back(case_paths);
		}else{
			cout << "Not exists:" + label_path << endl;
		}
	}
  return case_paths_list;
}

CaseObject readImageLabelCase( CasePaths case_paths, JSONObject *model_json, vector<json_token_t*> model_tokens )
{
  ImageProcessor *image_processor = new ImageProcessor();
  json_token_t* nueral_network = model_json->getChildForToken(model_tokens[0], "net");

  int classes = 1;
  int max_bounding_boxes = OBJECT_DETECTION_MAX_BOUNDBOXES;
  int image_w = OBJECT_DETECTION_IMAGE_WIDTH;
  int image_h = OBJECT_DETECTION_IMAGE_HEIGHT;

  string tmp = model_json->getChildValueForToken(nueral_network, "max_classes");
  if(tmp.length()>0){
    classes = std::stoi( tmp );
  }
  tmp = model_json->getChildValueForToken(nueral_network, "max_bounding_boxes");
  if(tmp.length()>0){
    max_bounding_boxes = std::stoi( tmp );
  }

  // cout << case_paths.image_path << endl;
  // cout << case_paths.label_path << endl;

  image_st image = image_processor->readImageFile(case_paths.image_path, image_w, image_h, 3);
  vector<boundingbox_t> boxes = readLabelBoxes(case_paths.label_path);

  CaseObject c {TensorObject<float>( 1, image_w, image_w, 3 ), TensorObject<float>( 1, max_bounding_boxes * (4 + classes), 1, 1 )};

  for(int z = 0; z < 3; ++z ){
    for ( int y = 0; y < image_h; y++ ){
      for ( int x = 0; x < image_w; x++ ){
        c.data( 0, x, y, z ) = image.data[x + (y * image_w) + (image_w * image_h)*z] / 255.f;
      }
    }
  }

  for( int j=0; j<max_bounding_boxes && j<boxes.size(); ++j){
    int index = j * (4 + classes);
    c.out( 0, index, 0, 0 )   = boxes[j].x;
    c.out( 0, index+1, 0, 0 ) = boxes[j].y;
    c.out( 0, index+2, 0, 0 ) = boxes[j].w;
    c.out( 0, index+3, 0, 0 ) = boxes[j].h;
    for ( int k = 0; k < classes; ++k ){
      c.out( 0, index+4+k, 0, 0 ) = boxes[j].id == k ? 1.0f : 0.0f;
    }
  }
  delete image_processor;
  return c;
}


vector<CaseObject> readCasesImagesLabels( JSONObject *data_json, vector<json_token_t*> data_tokens, JSONObject *model_json, vector<json_token_t*> model_tokens, string mode )
{
	Utils *utils = new Utils();
	vector<CaseObject> cases;
  vector<CasePaths> case_paths_list = listImageLabelCasePaths( data_json, data_tokens, model_json, model_tokens, mode );

  for(unsigned i=0; i<case_paths_list.size(); ++i){
    CaseObject c = readImageLabelCase( case_paths_list[i], model_json, model_tokens );
		cases.push_back( c );
	}

	delete utils;
	return cases;
}

vector<CaseObject> readCases(string data_json_path, string model_json_path, string mode) // dataset.cpp
{
	vector<CaseObject> cases;

	JSONObject *data_json = new JSONObject();
	JSONObject *model_json = new JSONObject();
	vector <json_token_t*> data_tokens = data_json->load(data_json_path);
	vector <json_token_t*> model_tokens = model_json->load(model_json_path);
	string dataset_type = data_json->getChildValueForToken(data_tokens[0], "type");   // data_tokens[0] := json root

	if(dataset_type=="mnist"){
		cases= readCasesMNIST(data_json, data_tokens, mode);
	}else if(dataset_type=="cifar10"){
		cases= readCasesCifar10(data_json, data_tokens, mode);
	}else if(dataset_type=="images_labels"){
		cases= readCasesImagesLabels(data_json, data_tokens, model_json, model_tokens, mode);
	}

	delete data_json;
	return cases;
}
