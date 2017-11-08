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
#ifndef FPGA_TOP_HPP
#define FPGA_TOP_HPP

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
#ifndef __SYNTHESIS__
	void fileOpen(layer_t &layer);
	void fileClose();
#endif
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
void loadWeightsFromDRAMKL(
        wemem_t *SHARED_DRAM,
        apfix32_weights_t WBRAM_KL[BLOCK_SIZE][PE_KL_CI][PE_KL_CO_PORT][SQR_KL_SIZE],
        apfix32_weights_t BBRAM_KL[BLOCK_SIZE][BIAS_NUM][WEMEM_1ADOTS_DIV8][PE_KL_CO_PORT]
	);
void dataFlowProcess(
        wemem_t *WEIGHTS_SHARED_DRAM,
        iemem_t *READ_SHARED_DRAM,
        iemem_t *WRITE_SHARED_DRAM
		);

void loadImageFromDRAM(
    iemem_t *READ_SHARED_DRAM   ,
    hls_pxl_strm_t &hlsPx5Strm
    );





void computeElement(
        hls_pxl_strm_t &hlsPx5Strm                                                   ,
//        hls_outl_strm_t &hlsOut5Strm                                                 ,
        apfix32_weights_t WBRAM_KL[BLOCK_SIZE][PE_KL_CI][PE_KL_CO_PORT][SQR_KL_SIZE]
        );
void getKLWeights(
        const channels_t coIndex                                                 ,
        const channels_t ciIndex                                                 ,
        apfix32_weights_t WBRAM[BLOCK_SIZE][PE_KL_CI][PE_KL_CO_PORT][SQR_KL_SIZE]   ,
        data_t weightsL[PE_KL_CO][PE_KL_CI][SQR_KL_SIZE]
        );
void macc2dKL(
        const data_t pixels5[SQR_KL_SIZE]   ,
        const data_t weights5[SQR_KL_SIZE]  ,
            result_t &result );

void macc2dKL_int8(
        const data_t pixelsL[SQR_KL_SIZE]   ,
        const data_t weightsLLow[SQR_KL_SIZE]  , const data_t weightsLHigh[SQR_KL_SIZE]  ,
            result_t &resultLow, result_t &resultHigh );

void macc2dKLInt8Tmp(
        const data_t pixels5[SQR_KL_SIZE]   ,
        const data_t weights5Low[SQR_KL_SIZE]  , const data_t weights5High[SQR_KL_SIZE]  ,
         result_t &resultLow, result_t &resultHigh);
#if 0
void writeBackToDRAM(
        hls_outl_strm_t &hlsOut5Strm,
        iemem_t *WRITE_SHARED_DRAM,
        apfix32_weights_t BBRAM_KL[BLOCK_SIZE][BIAS_NUM][WEMEM_1ADOTS_DIV8][PE_KL_CO_PORT]
    );

#endif


#endif
