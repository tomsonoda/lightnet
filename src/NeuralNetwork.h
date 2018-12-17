#pragma once
#include "CaseObject.h"
#include "TensorObject.h"
#include "LayerDense.h"
#include "LayerMaxPool.h"
#include "LayerReLU.h"
#include "LayerConvolution.h"
#include "LayerDropout.h"
#include "LayerSigmoid.h"
#include "LayerSoftmax.h"
#include "Types.h"

static void calc_grads( LayerObject* layer, TensorObject<float>& grad_next_layer )
{
	switch ( layer->type )
	{
		case LayerType::conv:
			((LayerConvolution*)layer)->calc_grads( grad_next_layer );
			return;
		case LayerType::dense:
			((LayerDense*)layer)->calc_grads( grad_next_layer );
			return;
		case LayerType::dropout:
			((LayerDropout*)layer)->calc_grads( grad_next_layer );
			return;
		case LayerType::relu:
			((LayerReLU*)layer)->calc_grads( grad_next_layer );
			return;
		case LayerType::max_pool:
			((LayerPool*)layer)->calc_grads( grad_next_layer );
			return;
		case LayerType::sigmoid:
			((LayerSigmoid*)layer)->calc_grads( grad_next_layer );
			return;
		case LayerType::softmax:
			((LayerSoftmax*)layer)->calc_grads( grad_next_layer );
			return;
		default:
			assert( false );
	}
}

static void fix_weights( LayerObject* layer )
{
	switch ( layer->type )
	{
		case LayerType::conv:
			((LayerConvolution*)layer)->fix_weights();
			return;
		case LayerType::dense:
			((LayerDense*)layer)->fix_weights();
			return;
		case LayerType::dropout:
			((LayerDropout*)layer)->fix_weights();
			return;
		case LayerType::max_pool:
			((LayerPool*)layer)->fix_weights();
			return;
		case LayerType::relu:
			((LayerReLU*)layer)->fix_weights();
			return;
		case LayerType::sigmoid:
			((LayerSigmoid*)layer)->fix_weights();
			return;
		case LayerType::softmax:
			((LayerSoftmax*)layer)->fix_weights();
			return;
		default:
			assert( false );
	}
}

static void activate( LayerObject* layer, TensorObject<float>& in )
{
	switch ( layer->type )
	{
		case LayerType::conv:
			((LayerConvolution*)layer)->activate( in );
			return;
		case LayerType::dense:
			((LayerDense*)layer)->activate( in );
			return;
		case LayerType::dropout:
			((LayerDropout*)layer)->activate( in );
			return;
		case LayerType::max_pool:
			((LayerPool*)layer)->activate( in );
			return;
		case LayerType::relu:
			((LayerReLU*)layer)->activate( in );
			return;
		case LayerType::sigmoid:
			((LayerSigmoid*)layer)->activate( in );
			return;
		case LayerType::softmax:
			((LayerSoftmax*)layer)->activate( in );
			return;
		default:
			assert( false );
	}
}

static vector<LayerObject*> loadModel(
	JSONObject *model_json,
	std::vector <json_token_t*> model_tokens,
	CaseObject case_object,
	float learning_rate)
{
  vector<LayerObject*> layers;

  std::vector<json_token_t*> json_layers = model_json->getArrayForToken(model_tokens[0], "layers");

  for(int i=0; i<json_layers.size(); i++){
    std::string type = model_json->getChildValueForToken(json_layers[i], "type");

    if(type=="convolutional"){

      uint16_t stride = std::stoi( model_json->getChildValueForToken(json_layers[i], "stride") );
      uint16_t size = std::stoi( model_json->getChildValueForToken(json_layers[i], "size") );
      uint16_t filters = std::stoi( model_json->getChildValueForToken(json_layers[i], "filters") );
      uint16_t padding = std::stoi( model_json->getChildValueForToken(json_layers[i], "padding") );
      TensorSize in_size;
      if(i==0){
        in_size = case_object.data.size;
      }else{
        in_size = layers[layers.size()-1]->out.size;
      }
			int out_size = (in_size.x - size + 2*padding)/stride + 1;
      printf("%d: convolutional batch (%d) : stride=%d  extend_filter=%d filters=%d pad=%d: (%d x %d x %d) -> ( %d x %d x %d)\n",
			i, in_size.b, stride, size, filters, padding, in_size.x, in_size.y, in_size.z, out_size, out_size, filters);
      LayerConvolution * layer = new LayerConvolution( stride, size, filters, padding, in_size);		// 28 * 28 * 1 -> 24 * 24 * 8
      layers.push_back( (LayerObject*)layer );

		}else if(type=="dense"){

			TensorSize in_size = ( i==0 ? (case_object.data.size) : (layers[layers.size()-1]->out.size) );
			int out_size=0;
			if(i==json_layers.size()-1){
				out_size = case_object.out.size.x * case_object.out.size.y * case_object.out.size.z;
			}else{
				out_size = std::stoi( model_json->getChildValueForToken(json_layers[i], "out_size") );
			}
      printf("%d: dense batch (%d) : ( %d x %d x %d ) -> ( %d ) \n",i, in_size.b, in_size.x, in_size.y, in_size.z, out_size);
      LayerDense *layer = new LayerDense(in_size, out_size, learning_rate);
      layers.push_back( (LayerObject*)layer );

		}else if(type=="maxpool"){

      uint16_t stride = std::stoi( model_json->getChildValueForToken(json_layers[i], "stride") );
      uint16_t size = std::stoi( model_json->getChildValueForToken(json_layers[i], "size") );
      TensorSize in_size = layers[layers.size()-1]->out.size;
			int out_size_x = (in_size.x - size ) / stride + 1;
			int out_size_y = (in_size.y - size ) / stride + 1;
      printf("%d: maxpool batch (%d) : stride=%d  extend_filter=%d: ( %d x %d x %d ) -> (%d x %d x %d)\n",
			i, in_size.b, stride, size, in_size.x, in_size.y, in_size.z, out_size_x, out_size_y, in_size.z);
      LayerPool * layer = new LayerPool( stride, size, in_size );				// 24 * 24 * 8 -> 12 * 12 * 8
      layers.push_back( (LayerObject*)layer );

    }else if(type=="relu"){

			TensorSize in_size = layers[layers.size()-1]->out.size;
      printf("%d: relu batch (%d) : ( %d x %d x %d ) -> ( %d x %d x %d ) \n", i, in_size.b, in_size.x, in_size.y, in_size.z, in_size.x, in_size.y, in_size.z);
      LayerReLU * layer = new LayerReLU( layers[layers.size()-1]->out.size );
      layers.push_back( (LayerObject*)layer );

		}else if (type=="route"){
			/*
      std::vector <json_token_t*> ref_layers = model_json->getArrayForToken(json_layers[i], "layers");
      for(int j=0; j<ref_layers.size(); j++){
        std::string value = model_json->getValueForToken(ref_layers[j]);
        printf("%s ", value.c_str());
      }
      printf("\n");
			*/
		}else if(type=="sigmoid"){

			TensorSize in_size = layers[layers.size()-1]->out.size;
			printf("%d: sigmoid batch (%d) : (%d) -> (%d)\n",i, in_size.b, (in_size.x * in_size.y * in_size.z), (in_size.x * in_size.y * in_size.z));
			LayerSigmoid *layer = new LayerSigmoid(in_size);
			layers.push_back( (LayerObject*)layer );

		}else if(type=="softmax"){

			TensorSize in_size = layers[layers.size()-1]->out.size;
			printf("%d: softmax batch (%d) : (%d) -> (%d)\n",i, in_size.b, (in_size.x * in_size.y * in_size.z), (in_size.x * in_size.y * in_size.z));
			LayerSoftmax *layer = new LayerSoftmax(in_size);
			layers.push_back( (LayerObject*)layer );

		}
  }
  return layers;
}

static void print_tensor( TensorObject<float>& data )
{
	int mx = data.size.x;
	int my = data.size.y;
	int mz = data.size.z;
	int mb = data.size.b;

	for ( int b = 0; b < mb; b++ ){
		printf( "[Batch %d]\n", b );
		for ( int z = 0; z < mz; z++ ){
			printf( "[Dim %d]\n", z );
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
