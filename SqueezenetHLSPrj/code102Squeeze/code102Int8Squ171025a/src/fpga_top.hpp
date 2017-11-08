//----------------------------------------------------------------
//  FPGA Accelerator For CNN Inference
//----------------------------------------------------------------
//
//  File:   fpga_top.hpp
//  CPU-Side Functions for FPGA Accelerator
//
//  (c) qiu chao, 2017-09
//
//----------------------------------------------------------------
#include "net_para.hpp"
#ifdef __SYNTHESIS__
	#include <ap_utils.h>
	#include <hls_math.h>
#endif
//================================================================
//=ExtMemToApFixSync
//================================================================
template <int N, typename T_SRC, typename T_DST, int W>
void ExtMemToApFixSync(T_SRC src, T_DST dst[W]) {
#pragma HLS inline
  for (int i = 0; i < W; i++) {
#pragma HLS UNROLL
    hls::AXIGetBitFields(src, i * N, N, dst[i]);
  }
}
//================================================================
//=ApFixToExtMemSync
//================================================================
template <int N, typename T_SRC, typename T_DST, int W>
void ApFixToExtMemSync(T_SRC &src, T_DST dst[W]) {
#pragma HLS inline
  for (int i = 0; i < W; i++) {
#pragma HLS UNROLL
    hls::AXISetBitFields(src, i * N, N, dst[i]);
  }
}
//================================================================
//=max
//================================================================
template <typename T_SRC> T_SRC max(T_SRC a, T_SRC b) {
#pragma HLS inline
  T_SRC MaxData;
  if (a >= b)
    MaxData = a;
  else
    MaxData = b;
  return MaxData;
}


//================================================================
//=Function definitions
//================================================================
void fpga_top(
        wemem_t *WEIGHTS_SHARED_DRAM,
        iemem_t *READ_SHARED_DRAM,
        iemem_t *WRITE_SHARED_DRAM,
        layer_t layer,
        weights_t nextWeightLayer,
		offset_t weightsOffset,
		offset_t imageOffset
        );
void setLayerConfig(
    layer_t &layer,
    weights_t &nextWeightLayer,
    offset_t weightsOffset,
    offset_t imageOffset
    );
void loadWeightsFromDRAM(
        wemem_t *SHARED_DRAM,
        apfix32_weights_t WBRAM_K5[BLOCK_SIZE][PE_K5_CO_PORT][PE_CI][SQR_K5_SIZE],
        apfix32_weights_t BBRAM_K5[BLOCK_SIZE][BIAS_NUM][WEMEM_1ADOTS_DIV8][PE_K5_CO_PORT]
	);
void loadImageFromDRAM(
    iemem_t *READ_SHARED_DRAM   ,
    hls_px5strm_t &hlsPx5Strm
    );
void computeElement(
        hls_px5strm_t &hlsPx5Strm                                                   ,
        hls_out5strm_t &hlsOut5Strm                                                 ,
        apfix32_weights_t WBRAM_K5[BLOCK_SIZE][PE_K5_CO_PORT][PE_CI][SQR_K5_SIZE]
        );
void getK5Weights(
        const channels_t coIndex                                                 ,
        const channels_t ciIndex                                                 ,
        apfix32_weights_t WBRAM[BLOCK_SIZE][PE_K5_CO_PORT][PE_CI][SQR_K5_SIZE]   ,   
        data_t weights5[PE_K5_CO][PE_CI][SQR_K5_SIZE]                         
        );
void macc2dK5(
        const data_t pixels5[SQR_K5_SIZE]   ,
        const data_t weights5[SQR_K5_SIZE]  ,
            result_t &result );

void macc2dK5_int8(
        const data_t pixels5[SQR_K5_SIZE]   ,
        const data_t weights5_C0[SQR_K5_SIZE]  , const data_t weights5_C1[SQR_K5_SIZE]  ,
            result_t &result_C0, result_t &result_C1 ) ;

void macc2dK5Int8Tmp(
        const data_t pixels5[SQR_K5_SIZE]   ,
        const data_t weights5Low[SQR_K5_SIZE]  , const data_t weights5High[SQR_K5_SIZE]  ,
         result_t &resultLow, result_t &resultHigh);

void writeBackToDRAM(
        hls_out5strm_t &hlsOut5Strm,
        iemem_t *WRITE_SHARED_DRAM,
        apfix32_weights_t BBRAM_K5[BLOCK_SIZE][BIAS_NUM][WEMEM_1ADOTS_DIV8][PE_K5_CO_PORT]
    );
void dataFlowProcess(
        wemem_t *WEIGHTS_SHARED_DRAM,
        iemem_t *READ_SHARED_DRAM,
        iemem_t *WRITE_SHARED_DRAM
		);

#ifndef __SYNTHESIS__
	void fileOpen(layer_t &layer);
	void fileClose();
#endif
