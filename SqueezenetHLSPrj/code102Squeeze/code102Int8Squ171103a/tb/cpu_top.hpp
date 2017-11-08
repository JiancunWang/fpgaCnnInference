//----------------------------------------------------------------
//  FPGA Accelerator For CNN Inference
//----------------------------------------------------------------
//
//  File:   cpu_top.hpp
//  CPU-Side Functions for FPGA Accelerator
//
//  (c) qiu chao, 2017-09
//
//----------------------------------------------------------------
#ifndef CPU_TOP_HPP
#define CPU_TOP_HPP
//================================================================
//=Standard C Libraries
//================================================================
#include <algorithm> // sort, reverse (on std::vector)
#include <cmath>     // fabs, fmax, ...
#include <cstdio>    // printf
#include <ctime>     // time() for random seed
#include <math.h>
#include <vector> // std::vector for softmax calculation
//================================================================
//=CNN Definitions
//================================================================
#include "hls_cm_log.hpp"
#include "network.hpp"
#include "../src/fpga_top.hpp"

struct softMaxResult_t {
	#define NAME_MAX_LEN  32
    char name[NAME_MAX_LEN+1];
    cpu_data_t value;
    softMaxResult_t(const char *n,cpu_data_t valueTmp):value(valueTmp){
        for (int i = 0; i < NAME_MAX_LEN; i++) {
          name[i] = n[i];
          if (n[i] == 0)
            break;
        }
        name[NAME_MAX_LEN] = 0;
    };
    softMaxResult_t():value(0) {name[0] = 0;};
public:
    void setName(const char *n){
        for (int i = 0; i < NAME_MAX_LEN; i++) {
          name[i] = n[i];
          if (n[i] == 0)
            break;
        }
        name[NAME_MAX_LEN] = 0;
    }
    void setValue(cpu_data_t valueTmp){
    	value = valueTmp;
    }
};

//================================================================
//=Function definitions
//================================================================
void loadInputImage(
        const char *fileName,
        cpu_data_t *imgInCpuAddr,
        layer_t *layer
        );
void transformInputImage(
    int multiTimes                  ,
    int inImgSize                   ,
    cpu_data_t  *imgInCpuAddr       ,
    layer_img_t *imgInBaseAddr        
    );
void loadInputFile(
        const char *fileName,
        cpu_data_t *inCpuAddr,
		int num);
void arrayMulAdd(
		cpu_data_t *iBuff,
		cpu_data_t *wBuff,
		cpu_data_t *bBuff,
		cpu_data_t *rBuff,
		bool logEn,
		int chIn,
		int chOut);

void hwcTransTochw(
		cpu_data_t *w0BuffIn,
		cpu_data_t *w0BuffOut,
		bool enIn,
		int lenIn,
		int chIn
	);
void softMax(
		cpu_data_t *rBuff,
		int chIn
		);
void innerProductProcess(
		cpu_data_t *iBuff,
		cpu_data_t *rBuff,
		int ch0 ,
		int ch1 ,
		int ch2
);

int main();


#endif
