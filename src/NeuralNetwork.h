#pragma once
#include "CaseObject.h"
#include "TensorObject.h"
#include "LayerBatchNormalization.h"
#include "LayerDense.h"
#include "LayerLeakyReLU.h"
#include "LayerMaxPool.h"
#include "LayerReLU.h"
#include "LayerRoute.h"
#include "LayerConvolution.h"
#include "LayerDropout.h"
#include "LayerSigmoid.h"
#include "LayerSoftmax.h"
#include "LayerType.h"
#include "ParameterObject.h"
#include "ThreadPool.h"

#define VERSION_MAJOR    (0)
#define VERSION_MINOR    (1)
#define VERSION_REVISION (0)

static void backward( LayerObject* layer, TensorObject<float>& dz_next_layer, ThreadPool& thread_pool )
{
	switch ( layer->type )
	{
		case LayerType::batch_normalization:
			((LayerBatchNormalization*)layer)->backward( dz_next_layer );
			return;
		case LayerType::conv:
			((LayerConvolution*)layer)->backward( dz_next_layer, thread_pool );
			return;
		case LayerType::dense:
			((LayerDense*)layer)->backward( dz_next_layer, thread_pool );
			return;
		case LayerType::dropout:
			((LayerDropout*)layer)->backward( dz_next_layer );
			return;
		case LayerType::leaky_relu:
			((LayerLeakyReLU*)layer)->backward( dz_next_layer );
			return;
		case LayerType::max_pool:
			((LayerPool*)layer)->backward( dz_next_layer );
			return;
		case LayerType::relu:
			((LayerReLU*)layer)->backward( dz_next_layer );
			return;
		case LayerType::route:
			((LayerRoute*)layer)->backward( dz_next_layer );
			return;
		case LayerType::sigmoid:
			((LayerSigmoid*)layer)->backward( dz_next_layer );
			return;
		case LayerType::softmax:
			((LayerSoftmax*)layer)->backward( dz_next_layer );
			return;
		default:
			assert( false );
	}
}

static void update_weights( LayerObject* layer )
{
	switch ( layer->type )
	{
		case LayerType::batch_normalization:
			((LayerBatchNormalization*)layer)->update_weights();
			return;
		case LayerType::conv:
			((LayerConvolution*)layer)->update_weights();
			return;
		case LayerType::dense:
			((LayerDense*)layer)->update_weights();
			return;
		case LayerType::dropout:
			((LayerDropout*)layer)->update_weights();
			return;
		case LayerType::leaky_relu:
				((LayerLeakyReLU*)layer)->update_weights();
				return;
		case LayerType::max_pool:
			((LayerPool*)layer)->update_weights();
			return;
		case LayerType::relu:
			((LayerReLU*)layer)->update_weights();
			return;
		case LayerType::route:
			((LayerRoute*)layer)->update_weights();
			return;
		case LayerType::sigmoid:
			((LayerSigmoid*)layer)->update_weights();
			return;
		case LayerType::softmax:
			((LayerSoftmax*)layer)->update_weights();
			return;
		default:
			assert( false );
	}
}

static void forward( LayerObject* layer, TensorObject<float>& in, ThreadPool& thread_pool )
{
	switch ( layer->type )
	{
		case LayerType::batch_normalization:
			((LayerBatchNormalization*)layer)->forward( in );
			return;
		case LayerType::conv:
			((LayerConvolution*)layer)->forward( in, thread_pool );
			return;
		case LayerType::dense:
			((LayerDense*)layer)->forward( in );
			return;
		case LayerType::dropout:
			((LayerDropout*)layer)->forward( in );
			return;
		case LayerType::leaky_relu:
				((LayerLeakyReLU*)layer)->forward( in );
				return;
		case LayerType::max_pool:
			((LayerPool*)layer)->forward( in );
			return;
		case LayerType::relu:
			((LayerReLU*)layer)->forward( in );
			return;
		case LayerType::route:
			((LayerRoute*)layer)->forward( in );
			return;
		case LayerType::sigmoid:
			((LayerSigmoid*)layer)->forward( in );
			return;
		case LayerType::softmax:
			((LayerSoftmax*)layer)->forward( in );
			return;
		default:
			printf("layer type=%d\n", layer->type);
			assert( false );
	}
}


static void saveWeights( LayerObject* layer, ofstream& fout )
{
	switch ( layer->type )
	{
		case LayerType::batch_normalization:
			((LayerBatchNormalization*)layer)->saveWeights( fout );
			return;
		case LayerType::conv:
			((LayerConvolution*)layer)->saveWeights( fout );
			return;
		case LayerType::dense:
			((LayerDense*)layer)->saveWeights( fout );
			return;
		case LayerType::dropout:
		case LayerType::leaky_relu:
		case LayerType::max_pool:
		case LayerType::relu:
		case LayerType::route:
		case LayerType::sigmoid:
		case LayerType::softmax:
			return;
		default:
			assert( false );
	}
}

static void saveLayersWeights( long step, vector<LayerObject*>& layers, string filename )
{
	std::ofstream fout( filename.c_str(), std::ios::binary );
	if (fout.fail()){
		std::cerr << "No weights file to save:" << filename << std::endl;
		return;
	}
	char ver_major    = (char)VERSION_MAJOR;
	char ver_minor    = (char)VERSION_MINOR;
	char ver_revision = (char)VERSION_REVISION;

	fout.write(( char * ) &(ver_major), sizeof( char ) );
	fout.write(( char * ) &(ver_minor), sizeof( char ) );
	fout.write(( char * ) &(ver_revision), sizeof( char ) );
	fout.write(( char * ) &(step), sizeof( long ) );

	for( int i = 0; i < layers.size(); ++i ){
		saveWeights( layers[i], fout );
	}
	fout.close();
}

static void loadWeights( LayerObject* layer, ifstream& fin )
{
	switch ( layer->type )
	{
		case LayerType::batch_normalization:
			((LayerBatchNormalization*)layer)->loadWeights( fin );
			return;
		case LayerType::conv:
			((LayerConvolution*)layer)->loadWeights( fin );
			return;
		case LayerType::dense:
			((LayerDense*)layer)->loadWeights( fin );
			return;
		case LayerType::dropout:
		case LayerType::leaky_relu:
		case LayerType::max_pool:
		case LayerType::relu:
		case LayerType::route:
		case LayerType::sigmoid:
		case LayerType::softmax:
			return;
		default:
			assert( false );
	}
}

static long loadLayersWeights( vector<LayerObject*>& layers, string filename )
{
	std::ifstream fin( filename.c_str(), std::ios::binary );

	if (fin.fail()){
		cout << "Error : Could not open a file to load:" << filename << std::endl;
		exit(0);
	}

	cout << "Loading weights from " << filename << " ..." << endl;
	char ver_major    = 0;
	char ver_minor    = 0;
	char ver_revision = 0;
	long step = 0;
	fin.read(( char * ) &(ver_major), sizeof( char ) );
	fin.read(( char * ) &(ver_minor), sizeof( char ) );
	fin.read(( char * ) &(ver_revision), sizeof( char ) );
	fin.read(( char * ) &(step), sizeof( long ) );
	cout << "Model version: " << to_string(ver_major) << "." << to_string(ver_minor) << "." << to_string(ver_revision) << endl;
	if( (ver_major!=VERSION_MAJOR) || (ver_minor!=VERSION_MINOR) || (ver_revision!=VERSION_REVISION) ){
		cout << "Model version is different from this version:" << to_string(VERSION_MAJOR) << "." << to_string(VERSION_MINOR) << "." << to_string(VERSION_REVISION) << endl;
		exit(0);
	}
	cout << "last step: " << to_string(step) << endl;

	for( int i = 0; i < layers.size(); ++i ){
		loadWeights( layers[i], fin );
	}
	fin.close();
	return step;
}


static vector<LayerObject*> loadModel(
	JSONObject *model_json,
	std::vector <json_token_t*> model_tokens,
	CaseObject case_object,
	float learning_rate,
	float decay,
	float momentum
	)
{
  vector<LayerObject*> layers;

  std::vector<json_token_t*> json_layers = model_json->getArrayForToken(model_tokens[0], "layers");

  for(int i=0; i<json_layers.size(); ++i){
    std::string type = model_json->getChildValueForToken(json_layers[i], "type");

		if(type=="batch_normalization"){
			TensorSize in_size;
			if(i==0){
				in_size = case_object.data.size;
			}else{
				in_size = layers[layers.size()-1]->out.size;
			}
			printf("%d: batch normalization: ( %d x %d x %d ) -> ( %d x %d x %d ) \n", i, in_size.x, in_size.y, in_size.z, in_size.x, in_size.y, in_size.z);
			LayerBatchNormalization *layer = new LayerBatchNormalization(in_size, learning_rate);
			layers.push_back( (LayerObject*)layer );

    }else if(type=="convolutional"){

			uint16_t stride = 1;
      uint16_t size = 5;
      uint16_t filters = 1;
      uint16_t padding = 0;

			string tmp = model_json->getChildValueForToken(json_layers[i], "stride") ;
			if(tmp.length()>0){
      	stride = std::stoi( tmp );
			}
			tmp = model_json->getChildValueForToken(json_layers[i], "size");
			if(tmp.length()>0){
				size = std::stoi( tmp );
			}
			tmp = model_json->getChildValueForToken(json_layers[i], "filters");
			if(tmp.length()>0){
				filters = std::stoi( tmp );
			}
			tmp = model_json->getChildValueForToken(json_layers[i], "padding");
			if(tmp.length()>0){
				padding = std::stoi( tmp );
			}

      TensorSize in_size;
      if(i==0){
        in_size = case_object.data.size;
      }else{
        in_size = layers[layers.size()-1]->out.size;
      }
			int out_size = (in_size.x - size + 2*padding)/stride + 1;
      printf("%d: convolutional      : stride=%d  kernel_size=%d filters=%d padding=%d: ( %d x %d x %d ) -> ( %d x %d x %d )\n",
			i, stride, size, filters, padding, in_size.x, in_size.y, in_size.z, out_size, out_size, filters);
      LayerConvolution * layer = new LayerConvolution( stride, size, filters, padding, in_size, learning_rate, decay, momentum);		// 28 * 28 * 1 -> 24 * 24 * 8

      layers.push_back( (LayerObject*)layer );

		}else if(type=="dense"){

			TensorSize in_size = ( i==0 ? (case_object.data.size) : (layers[layers.size()-1]->out.size) );
			int out_size=0;
			if(i==json_layers.size()-1){
				out_size = case_object.out.size.x * case_object.out.size.y * case_object.out.size.z;
			}else{
				out_size = std::stoi( model_json->getChildValueForToken(json_layers[i], "out_size") );
			}
      printf("%d: dense              : ( %d x %d x %d ) -> ( %d ) \n",i, in_size.x, in_size.y, in_size.z, out_size);
      LayerDense *layer = new LayerDense(in_size, out_size, learning_rate, decay, momentum);
      layers.push_back( (LayerObject*)layer );

		}else if (type=="leaky_relu"){

			TensorSize in_size = layers[layers.size()-1]->out.size;
      printf("%d: leaky relu         : ( %d x %d x %d ) -> ( %d x %d x %d ) \n", i, in_size.x, in_size.y, in_size.z, in_size.x, in_size.y, in_size.z);
      LayerLeakyReLU *layer = new LayerLeakyReLU( layers[layers.size()-1]->out.size );
      layers.push_back( (LayerObject*)layer );

		}else if(type=="maxpool"){

      uint16_t stride = std::stoi( model_json->getChildValueForToken(json_layers[i], "stride") );
      uint16_t size = std::stoi( model_json->getChildValueForToken(json_layers[i], "size") );
      TensorSize in_size = layers[layers.size()-1]->out.size;
			int out_size_x = (in_size.x - size ) / stride + 1;
			int out_size_y = (in_size.y - size ) / stride + 1;
      printf("%d: maxpool            : stride=%d  kernel_size=%d: ( %d x %d x %d ) -> ( %d x %d x %d )\n",
			i, stride, size, in_size.x, in_size.y, in_size.z, out_size_x, out_size_y, in_size.z);
      LayerPool * layer = new LayerPool( stride, size, in_size );				// 24 * 24 * 8 -> 12 * 12 * 8
      layers.push_back( (LayerObject*)layer );

    }else if(type=="relu"){

			TensorSize in_size = layers[layers.size()-1]->out.size;
      printf("%d: relu : ( %d x %d x %d ) -> ( %d x %d x %d ) \n", i, in_size.x, in_size.y, in_size.z, in_size.x, in_size.y, in_size.z);
      LayerReLU *layer = new LayerReLU( layers[layers.size()-1]->out.size );
      layers.push_back( (LayerObject*)layer );

		}else if (type=="route"){

			vector <json_token_t*> json_ref_layers = model_json->getArrayForToken(json_layers[i], "layers");
			vector <int> ref_layers;
			int b_val = 0;
			int x_val = 0;
			int y_val = 0;
			int z_sum = 0;
			printf("%d: route              : [", i);
      for(int j=0; j<json_ref_layers.size(); ++j){
        string value_str = model_json->getValueForToken(json_ref_layers[j]);
				if(value_str.size()>0){
					uint16_t ref_index = std::stoi( value_str ) + i;
					printf("%d ", ref_index);
					ref_layers.push_back(ref_index);
					b_val = layers[ref_index]->out.size.b;
					x_val = layers[ref_index]->out.size.x;
					y_val = layers[ref_index]->out.size.y;
					z_sum += layers[ref_index]->out.size.z;
    		}
      }
			printf("] -> ( %d x %d x %d )\n", x_val, y_val, z_sum);

			if(ref_layers.size()>0){
				for(int j=0; j<ref_layers.size(); ++j){
					if(x_val != layers[ref_layers[j]]->out.size.x){
						printf("reference layer x-sizes are different.\n");
						exit(0);
					}
					if(y_val != layers[ref_layers[j]]->out.size.y){
						printf("reference layer y-sizes are different.\n");
						exit(0);
					}
				}
				LayerRoute *layer = new LayerRoute( layers, ref_layers, {b_val, x_val, y_val, z_sum});
				layers.push_back( (LayerObject*)layer );
			}

		}else if(type=="sigmoid"){

			TensorSize in_size = layers[layers.size()-1]->out.size;
			printf("%d: sigmoid            : ( %d ) -> ( %d )\n",i, (in_size.x * in_size.y * in_size.z), (in_size.x * in_size.y * in_size.z));
			LayerSigmoid *layer = new LayerSigmoid(in_size);
			layers.push_back( (LayerObject*)layer );

		}else if(type=="softmax"){

			TensorSize in_size = layers[layers.size()-1]->out.size;
			printf("%d: softmax            : ( %d ) -> ( %d )\n",i, (in_size.x * in_size.y * in_size.z), (in_size.x * in_size.y * in_size.z));
			LayerSoftmax *layer = new LayerSoftmax(in_size);
			layers.push_back( (LayerObject*)layer );

		}else{

			assert( false );

		}
  }
  return layers;
}


static void loadModelParameters(JSONObject *model_json, vector <json_token_t*> model_tokens, ParameterObject* parameter_object)
{
	json_token_t* nueral_network = model_json->getChildForToken(model_tokens[0], "net");

	string tmp = model_json->getChildValueForToken(nueral_network, "batch_size");
	if(tmp.length()>0){
		parameter_object->batch_size = std::stoi( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "threads");
	if(tmp.length()>0){
		parameter_object->threads = std::stoi( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "learning_rate");
	if(tmp.length()>0){
		parameter_object->learning_rate = std::stof( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "momentum");
	if(tmp.length()>0){
		parameter_object->momentum = std::stof( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "weights_decay");
	if(tmp.length()>0){
		parameter_object->weights_decay = std::stof( tmp );
	}
	tmp = model_json->getChildValueForToken(nueral_network, "optimization");
	if(tmp.length()>0){
		parameter_object->optimizer = tmp;
	}
	tmp = model_json->getChildValueForToken(nueral_network, "train_output_span");
	if(tmp.length()>0){
		parameter_object->train_output_span = std::stoi( tmp );
	}

	if(parameter_object->batch_size<0){
		fprintf(stderr, "Batch size should be 1>=.");
		exit(0);
	}
}

static void printTensor( TensorObject<float>& data )
{
	int mx = data.size.x;
	int my = data.size.y;
	int mz = data.size.z;
	int mb = data.size.b;

	for ( int b = 0; b < mb; ++b ){
		printf( "[Batch %d]\n", b );
		for ( int z = 0; z < mz; ++z ){
			// printf( "[Dim %d]\n", z );
			for ( int y = 0; y < my; y++ ){
				for ( int x = 0; x < mx; x++ ){
					printf( "%.3f \t", (float)data( b, x, y, z ) );
				}
				printf( "\n" );
			}
			printf( "\n" );
		}
	}
}
