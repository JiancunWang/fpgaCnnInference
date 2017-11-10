//----------------------------------------------------------------
//  FPGA Accelerator For CNN Inference
//----------------------------------------------------------------
//
//  File:   network.cpp
//  CPU-Side Functions for FPGA Accelerator
//
//  (c) qiu chao, 2017-09
//
//----------------------------------------------------------------
#ifndef NETWORK_HPP
#define NETWORK_HPP
#include "../src/net_para.hpp"
#include "hls_cm_log.hpp"
//================================================================
//=Network type definition
//================================================================
struct network_t {
    layer_t *layers;
    num_layers_t numLayers;
    layer_weights_t *weightsBaseAddr;
    num_weights_bytes_t numWeightsBytes;
    num_image_bytes_t numImageBytes; 
    // default constructor: need to give max_layers and max_weightss
    // allocates layers[max_layers] and weightss[max_weightss] on Heap
    // -> can only be used on CPU, not on FPGA
    network_t(num_layers_t maxLayers, num_weights_bytes_t maxWeightsBytes){
        layers = (layer_t *)malloc( (sizeof(layer_t))  * maxLayers);
        weightsBaseAddr = (layer_weights_t *)malloc(maxWeightsBytes);
        numWeightsBytes= maxWeightsBytes;
        //initial others in the next function
        numLayers = 0;
        numImageBytes=0;
    }
};
//================================================================
//=Function definitions
//================================================================
network_t *getNetWorkConfig(
    );
void addLayer(
    network_t *net, 
    layer_t layer
    );
void loadWeightsFromFile(
    network_t *net,
    const char *fileName
    );
void transformWeightsData(
    FILE *fileHandle,
    int numWeights,
    cpu_data_t weightsTimes,
    cpu_data_t *weightsCpuAddr,
    layer_weights_t *weightsLayerAddr,
    layer_t *layer
    );
void transformCommonBiasData(
    FILE *fileHandle,
    int numBias,
    cpu_data_t biasTimes,
    cpu_data_t *biasCpuAddr,
    layer_weights_t *biasLayerAddr
    );
#endif
