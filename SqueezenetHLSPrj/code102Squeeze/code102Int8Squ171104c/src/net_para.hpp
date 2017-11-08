//----------------------------------------------------------------
//  FPGA Accelerator For CNN Inference
//----------------------------------------------------------------
//
//  File:   net_para.hpp 
//  FPGA-Side network and config parameters  
//
//  (c) qiu chao, 2017-09
//
//----------------------------------------------------------------
#ifndef NET_PARA_HPP
#define NET_PARA_HPP
#include "ap_int.h"
#include <hls_video.h>
#include <cmath>  
#include <cstdlib>
#include <string>
#include <stdint.h>

//================================================================
//=Bit-Width Calculation Macros
//================================================================
// NBITS(constant) = how many bits needed to represent <constant>
#define NBITS2(n) ((n & 2) ? 1 : 0)
#define NBITS4(n) ((n & (0xC)) ? (2 + NBITS2(n >> 2)) : (NBITS2(n)))
#define NBITS8(n) ((n & 0xF0) ? (4 + NBITS4(n >> 4)) : (NBITS4(n)))
#define NBITS16(n) ((n & 0xFF00) ? (8 + NBITS8(n >> 8)) : (NBITS8(n)))
#define NBITS32(n) ((n & 0xFFFF0000) ? (16 + NBITS16(n >> 16)) : (NBITS16(n)))
#define NBITS(n) ((n) == 0 ? 1 : NBITS32((n)) + 1)
//================================================================
//=CEIL_DIV
//================================================================
// up to integer, why should "-1"
// case x == 0          --> (((x)+(y)-1)/(y)) = (y-1)/y     = 0
// case x == 1,2,3,...y --> (((x)+(y)-1)/(y)) = (y+y-1)/y   = 1
// case x == y+1        --> (((x)+(y)-1)/(y)) = (y+1+y-1)/y = 2 
#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))
//================================================================
//=FLOOR_DIV
//================================================================
// up to integer, why should "-1"
#define FLOOR_DIV(x,y) ((x)/(y))
//================================================================
//=MAX_DOT
//================================================================
#define MAX_DOT(x,y) ((x) > (y) ? (x) : (y))
//================================================================
//=Const Variables
//================================================================
const int TOTAL_NUM_LAYERS = 26;
const long TOTAL_WEIGHT_BYTES = 1248424;
//const long TOTAL_DRAM_BYTES = 53248;
//const long DRAM_DEPTH = (TOTAL_WEIGHT_BYTES+TOTAL_DRAM_BYTES) / IEMEM_1ADOTS;
const long DRAM_DEPTH = 380747;
//const int MAX_WEIGHTS_PER_LAYER = 1;
// image size 512x512
const int MAX_DIMENSION = 512;
// Max(co x width)
const int MAX_IMAGE_CACHE_SIZE = 65536;
const int MAX_NUM_CHOUT = 1024;
const int MAX_NUM_SPLIT_CHOUT = 256;
const int MAX_CHANNELS  = 1024;
const int MAX_LINE_NUM  = 8192;
const int KL_SIZE       = 11;//11x11,7x7,5x5
const int KS_SIZE       = 3; //3x3,1x1
const int INT8_TIMES = 8;
const int MAX_KERNEL_SIZE = MAX_DOT(KL_SIZE,KS_SIZE);
const int LOG2_NUM_IMG_CACHE_LINES = NBITS(MAX_KERNEL_SIZE);
const int NUM_IMG_CACHE_LINES = 1 << LOG2_NUM_IMG_CACHE_LINES;

const int MAX_FILTER_SIZE = MAX_KERNEL_SIZE*MAX_KERNEL_SIZE; 
const int MAX_STRIDE_SIZE = 2;
const int MAX_POOLING_SIZE = 2;
const int MAX_SUBLAYER_NUM = 1024;
const int BRAM_WIDTH = 32;
const int BLOCK_SIZE = 512; 
const int SQR_KS_SIZE = KS_SIZE * KS_SIZE; 
const int SQR_KL_SIZE = KL_SIZE * KL_SIZE;
const int PE_KS_CO = 64; 
const int PE_KS_CI = 4; 
const int PE_KL_CO = 8;
const int PE_KL_CI = 1;
const int MAX_TIMES_SIZE=SQR_KL_SIZE/INT8_TIMES+1;
const int  ACCU_GROUP=( SQR_KL_SIZE%INT8_TIMES )?( SQR_KL_SIZE/INT8_TIMES+1 ):( SQR_KL_SIZE/INT8_TIMES );
// Int 8 Size
const int DOT_BYTES = 1; 
const int DOT_BITS = 8; 
const int MAX_MULTI_TIMES = 256;
// IEMEM means "image extern memory"
const int IEMEM_WIDTH = 128;
const int HALF_IEMEM_WIDTH = IEMEM_WIDTH / 2; 
// WEMEM means "weights extern memory"
const int WEMEM_WIDTH = 128;
// IEMEM_1ADOTS means "How many dots are stored in one address of image extern memory"
const int IEMEM_1ADOTS = IEMEM_WIDTH / DOT_BITS;//128/8=16
// WEMEM_1ADOTS means "How many dots are stored in one address of weights extern memory"
const int WEMEM_1ADOTS = WEMEM_WIDTH / DOT_BITS;//128/8=16
const int HALF_IEMEM_1ADOTS = IEMEM_1ADOTS / 2;//16/2=8
const int HALF_WEMEM_1ADOTS = WEMEM_1ADOTS / 2;//16/2=8
const int SIZEOF_IEMEM = IEMEM_WIDTH / DOT_BITS;//128/8=16
const int SIZEOF_WEMEM = WEMEM_WIDTH / DOT_BITS;//128/8=16
const int EMEM_1ADOTS = MAX_DOT(IEMEM_1ADOTS,WEMEM_1ADOTS);
// BRAM_1ADOTS means "How many dots are stored in one address of bram".
const int BRAM_1ADOTS = BRAM_WIDTH/DOT_BITS;//32/8=4
const int NET_NAME_MAX_LEN = 10; 
// 8bit x 4 x 9 = 288 
const int PXS_STRM_WIDTH = KS_SIZE * KS_SIZE * BRAM_1ADOTS * DOT_BITS;//288
// 8bit x 121 = 968
const int PXL_STRM_WIDTH = KL_SIZE * KL_SIZE * DOT_BITS;//968
// 8bit x 2 x PE_KS_CO  
const int OUTS_STRM_WIDTH = DOT_BITS * 2 * PE_KS_CO;//1024
// 8bit x 2 x  PE_KL_CO , why 2, because result = 8bit x 8bit = 16bit
const int OUTL_STRM_WIDTH = DOT_BITS * 2 * PE_KL_CO;//128
// align data in DRAM to 4KB borders
const int MEMORY_ALIGNMENT = SIZEOF_IEMEM * 1024; 
const int BIAS_NUM = 1;
const int PE_KL_CO_PORT = PE_KL_CO/BRAM_1ADOTS;//2
const int IEMEM_1ADOTS_DIV4 = IEMEM_1ADOTS/BRAM_1ADOTS;//16/4
const int WEMEM_1ADOTS_DIV4 = WEMEM_1ADOTS/BRAM_1ADOTS;//16/4
const int WEMEM_1ADOTS_DIV8 = WEMEM_1ADOTS/(BRAM_1ADOTS*PE_KL_CO_PORT);//2
const int MAX_WIDTH_X_CHIN_DIV16_KL = MAX_DIMENSION*IEMEM_1ADOTS/PE_KL_CI;//1024x16 dots(for int8)
const int MAX_WIDTH_X_CHIN_DIV16 = 1024;//1024x16 dots(for int8)
// Mean Pixel for ImageNet Data
const float MEAN_R = 104;
const float MEAN_G = 117;
const float MEAN_B = 123;
const int MAX_WBUFF_BYTES = DOT_BYTES * MAX_DOT(MAX_NUM_SPLIT_CHOUT * MAX_FILTER_SIZE , MAX_NUM_SPLIT_CHOUT * BIAS_NUM );
const int MAX_WBUFF_SIZE = MAX_DOT(MAX_WBUFF_BYTES / WEMEM_1ADOTS , BLOCK_SIZE);
//================================================================
//=Type-Definitions
//================================================================
//ap_int and ap_uint
//typedef ap_int<NBITS(MAX_DIMENSION+MAX_KERNEL_SIZE)> dimension_t;
typedef short dimension_t;
typedef ap_uint<NBITS(MAX_CHANNELS)> channels_t;
// typedef ap_uint<NBITS(MAX_WEIGHTS_PER_LAYER)> per_num_weights_t;
typedef ap_uint<NBITS(MAX_SUBLAYER_NUM)> num_sublayers_t;
typedef ap_uint<NBITS(TOTAL_NUM_LAYERS)> num_layers_t;
typedef ap_uint<NBITS(MAX_KERNEL_SIZE)> kernel_t;
typedef ap_uint<NBITS(MAX_FILTER_SIZE)> filter_t;
typedef ap_uint<NBITS(MAX_STRIDE_SIZE)> stride_t; 
typedef ap_uint<NBITS(MAX_POOLING_SIZE)> pooling_t; 
typedef ap_int<NBITS(MAX_MULTI_TIMES)> pos_t;
typedef ap_uint<BRAM_WIDTH> apfix32_weights_t;
typedef ap_uint<BRAM_WIDTH> apfix32_image_t;
typedef ap_uint<NBITS(IEMEM_1ADOTS)> bias_t;
typedef ap_uint<NBITS(EMEM_1ADOTS)> emem_1adots_t;
typedef ap_uint<NBITS(BLOCK_SIZE)> block_size_t;
typedef ap_uint<40> memaddr_t;
typedef ap_uint<IEMEM_WIDTH> iemem_t;
typedef ap_uint<HALF_IEMEM_WIDTH> half_iemem_t;
typedef ap_uint<WEMEM_WIDTH> wemem_t;
typedef ap_uint<PXS_STRM_WIDTH> pxs_strm_t;
typedef ap_uint<PXL_STRM_WIDTH> pxl_strm_t;
typedef ap_uint<OUTS_STRM_WIDTH> outs_strm_t;
typedef ap_uint<OUTL_STRM_WIDTH> outl_strm_t;
typedef ap_uint<NBITS(MAX_TIMES_SIZE)> accu_group_t;
typedef ap_uint<6> pad_size_t;
typedef ap_int<8> data_t;
typedef short data_result_t;
typedef ap_int<DOT_BITS+DOT_BITS+NBITS(SQR_KL_SIZE)> result_t;
typedef hls::stream<pxs_strm_t> hls_pxs_strm_t;
typedef hls::stream<pxl_strm_t> hls_pxl_strm_t;
typedef hls::stream<outs_strm_t> hls_outs_strm_t;
typedef hls::stream<outl_strm_t> hls_outl_strm_t;
typedef hls::Window<KL_SIZE,KL_SIZE,data_t> win_pix_kl_t;
typedef hls::Window<KS_SIZE,KS_SIZE,data_t> win_pix_ks_t;
//misc
typedef char layer_weights_t;
typedef char layer_img_t;
typedef int num_weights_bytes_t;
typedef int num_image_bytes_t;
typedef float cpu_data_t  ;
typedef uint8_t uatom_t;
typedef char atom_t;
typedef unsigned int offset_t;
typedef  ap_int<27>  apfix27_weight_t;
typedef  ap_int<42>  apfix45_dsp_out_t;

//================================================================
//=int8 def
//================================================================
#define BIT_16_TO_0      0x000000001ffff
#define BIT_47_TO_17     0xfffffffe0000
#define BIT_17           0x000000020000
#define BIT_35           0x000800000000


//================================================================
//=weights_t
//================================================================
struct weights_t{
    num_layers_t layerNum;
    channels_t channelsIn;
    channels_t channelsOut;
    kernel_t kernel;
    // per_num_weights_t weightsNum;
    memaddr_t memAddrWeights;
    memaddr_t layerBiasOffset[BIAS_NUM];
    weights_t()
        :layerNum(0),channelsIn(0),channelsOut(0),
        kernel(0),memAddrWeights(0){
            for(int i=0;i<BIAS_NUM;i++){
                layerBiasOffset[i] = 0;
            }
        };
};
//================================================================
//=layer_t
//================================================================
struct layer_t{
    char name[NET_NAME_MAX_LEN+1];
    dimension_t width ;
    dimension_t height;
    channels_t channelsIn;
    channels_t channelsOut;
    kernel_t kernel;
    pad_size_t padSize;
    stride_t stride;
    bool relu;
    pooling_t globalPooling;////==0,no pooling , ==1,pooling size 1, ==2,pooling size 2
    bool batchNorm;
    num_sublayers_t sublayerNum; 
    num_sublayers_t sublayerSeq;
    pos_t imgPosIn;
    pos_t imgPosOut;
    pos_t weightsPos;
    pos_t biasPos; 
    pos_t scalesPos;
    pos_t meanPos;
    pos_t variancePos;
    memaddr_t memAddrBias;
    memaddr_t memAddrScale;
    memaddr_t memAddrRollingMean;
    memaddr_t memAddrRollingVariance;
    memaddr_t memAddrWeights;
    memaddr_t memAddrImgIn;
    memaddr_t memAddrImgOut;
    // full constructor, used to define network in network.cpp
    layer_t(const char *n, int w, int h, int ci, int co, int k, int pSize, int s, 
          int r,
          int globalPool = 0, bool batch= 0, 
          int sublayerNumber = 0, int sublayerSeqNum = 0,
          int iposIn = 0, int iposOut=0,int wpos = 0,
          int bpos = 0, int spos = 0,int mpos = 0,int vpos = 0,
          int mem_b = 0, int mem_s = 0, int mem_m = 0,int mem_v = 0,
          int mem_w = 0, int mem_i = 0, int mem_o = 0)
      : width(w), 
        height(h), 
        channelsIn(ci), 
        channelsOut(co), 
        kernel(k),
        padSize(pSize),
        stride(s),
        relu(r),
        globalPooling(globalPool),
        batchNorm(batch),
        sublayerNum(sublayerNumber),
        sublayerSeq(sublayerSeqNum),
        imgPosIn(iposIn),
        imgPosOut(iposOut),
        weightsPos(wpos),
        biasPos(bpos),
        scalesPos(spos),
        meanPos(mpos),
        variancePos(vpos),
        memAddrBias(mem_b),
        memAddrScale(mem_s),
        memAddrRollingMean(mem_m),
        memAddrRollingVariance(mem_v),
        memAddrWeights(mem_w),
        memAddrImgIn(mem_i),
        memAddrImgOut(mem_o) {
        for (int i = 0; i < NET_NAME_MAX_LEN; i++) {
          name[i] = n[i];
          if (n[i] == 0)
            break;
        }
        name[NET_NAME_MAX_LEN] = 0;
    };
    layer_t()
      : width(0), 
        height(0), 
        channelsIn(0), 
        channelsOut(0), 
        kernel(0),
        padSize(0),
        stride(0),
        relu(0),
        globalPooling(0),
        batchNorm(0),
        sublayerNum(0),
        sublayerSeq(0),
        imgPosIn(0),
        imgPosOut(0),
        weightsPos(0),
        biasPos(0),
        scalesPos(0),
        meanPos(0),
        variancePos(0),
        memAddrBias(0),
        memAddrScale(0),
        memAddrRollingMean(0),
        memAddrRollingVariance(0),
        memAddrWeights(0),
        memAddrImgIn(0),
        memAddrImgOut(0) {
        name[0] = 0;
        };
};

#endif
