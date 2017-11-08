//----------------------------------------------------------------
//  FPGA Accelerator For CNN Inference
//----------------------------------------------------------------
//
//  File:   fpga_top.cpp
//  CPU-Side Functions for FPGA Accelerator
//
//  (c) qiu chao, 2017-09
//
//----------------------------------------------------------------
#include "fpga_top.hpp"
//================================================================
//= Global Variables
//================================================================
//-----------------------------[512]-------[25]-----[9]------[4]
//apfix32_weights_t WEIGHTS_RAM_A[BLOCK_SIZE][SQR_K5_SIZE][K3_SIZE][PE_CI];
//-------------------------[512]-------[2]------------[4]----[25]
apfix32_weights_t WBRAM_K5A[BLOCK_SIZE][PE_K5_CO_PORT][PE_CI][SQR_K5_SIZE];
//-------------------------[512]-------[4]-------[2]                [2]
apfix32_weights_t BBRAM_K5A[BLOCK_SIZE][BIAS_NUM][WEMEM_1ADOTS_DIV8][PE_K5_CO_PORT];

//----------------------[512]					[8]					 [4]
apfix32_image_t IBRAM_K5[MAX_WIDTH_X_CHIN_DIV16][NUM_IMG_CACHE_LINES][PE_CI];
//----------[512]					[16]
data_t LBRAM[MAX_WIDTH_X_CHIN_DIV16*2][PE_K5_CO];////512x16
#define MAX_SHORT 16383             ; 
memaddr_t WEIGHTS_OFFSET            ;
memaddr_t LAYER_WEIGHTS_OFFSET      ;
memaddr_t LAYER_BIAS_OFFSET         ;
memaddr_t IMAGE_OFFSET              ;
memaddr_t LAYER_IN_IMAGE_OFFSET     ;
memaddr_t LAYER_OUT_IMAGE_OFFSET    ;
channels_t DDR_CH_IN                ;
channels_t CH_IN                    ; 
channels_t CH_OUT                   ; 
dimension_t WIDTH_IN                ;
dimension_t HEIGHT_IN               ;
dimension_t WIDTH_OUT               ;
dimension_t HEIGHT_OUT              ;
dimension_t LOOP_CNT                ;
pooling_t POOLING_TYPE              ;
num_layers_t LAYER_NUM              ; 
kernel_t KERNEL                     ;
stride_t STRIDE                     ;
pad_size_t PAD_SIZE                 ;
bool RELU                           ;
bool IS_FIRST_SPLIT_LAYER           ;
bool IS_SECOND_SPLIT_LAYER          ;
bool BATCH_NORM                     ;
num_sublayers_t SUB_LAYER_NUM       ;
num_sublayers_t SUB_LAYER_SEQ       ;
num_sublayers_t SUB_LAYER_FLAG      ;
pos_t IMG_POS_IN                    ;
pos_t IMG_POS_OUT                   ;
pos_t WEIGHTS_POS                   ;
pos_t BIAS_POS                      ;
pos_t SCALES_POS                    ;
pos_t MEAN_POS                      ;
pos_t VARIANCE_POS                  ;
channels_t NEXT_CH_IN               ;
channels_t NEXT_CH_OUT              ;
kernel_t NEXT_KERNEL                ;
memaddr_t NEXT_NUM_WEIGHTS          ;
memaddr_t NEXT_LAYER_WEIGHTS_OFFSET ;
num_layers_t NEXT_LAYER_NUM			;
//================================================================
//= fpga_top
//================================================================
void fpga_top(
        wemem_t *WEIGHTS_SHARED_DRAM,
        iemem_t *READ_SHARED_DRAM,
        iemem_t *WRITE_SHARED_DRAM,
        layer_t layer,
        weights_t nextWeightLayer,
		offset_t weightsOffset,
		offset_t imageOffset
        ){
#pragma HLS INTERFACE m_axi depth = DRAM_DEPTH port=WEIGHTS_SHARED_DRAM offset = slave bundle = memorybus0 register
#pragma HLS INTERFACE m_axi depth = DRAM_DEPTH port=READ_SHARED_DRAM offset = slave bundle = memorybus1 register
#pragma HLS INTERFACE m_axi depth = DRAM_DEPTH port=WRITE_SHARED_DRAM offset = slave bundle = memorybus2 register
#pragma HLS INTERFACE s_axilite port = layer bundle = axilite register
#pragma HLS INTERFACE s_axilite port = nextWeightLayer bundle = axilite register
#pragma HLS INTERFACE s_axilite port = weightsOffset bundle = axilite register
#pragma HLS INTERFACE s_axilite port = imageOffset bundle = axilite register
#pragma HLS INTERFACE s_axilite port = return bundle = axilite register
    #ifndef __SYNTHESIS__
        fileOpen(layer);
        printf("CNN Inference FPGA Accelerating\n");
    #endif



    setLayerConfig(layer,nextWeightLayer,weightsOffset,imageOffset);


    loadWeightsFromDRAM(
    		WEIGHTS_SHARED_DRAM,
			WBRAM_K5A,
			BBRAM_K5A);

    dataFlowProcess(
    		WEIGHTS_SHARED_DRAM,
			READ_SHARED_DRAM,
			WRITE_SHARED_DRAM);



    #ifndef __SYNTHESIS__
        fileClose();
    #endif


}
//================================================================
//= setLayerConfig 
//================================================================
void setLayerConfig(
    layer_t &layer,
    weights_t &nextWeightLayer,
    offset_t weightsOffset,
    offset_t imageOffset
    ){
    WEIGHTS_OFFSET = (weightsOffset);
    LAYER_WEIGHTS_OFFSET = layer.memAddrWeights/sizeof(wemem_t);
    LAYER_BIAS_OFFSET = layer.memAddrBias/sizeof(wemem_t);

    IMAGE_OFFSET = (imageOffset);
    LAYER_IN_IMAGE_OFFSET  = layer.memAddrImgIn/sizeof(iemem_t);
    LAYER_OUT_IMAGE_OFFSET = layer.memAddrImgOut/sizeof(iemem_t);
    // layer.channelsIn must be 16 times
    if(layer.channelsIn == 3){
        DDR_CH_IN = IEMEM_1ADOTS;
    }else{
        DDR_CH_IN = layer.channelsIn    ; 
    }
    CH_IN = layer.channelsIn ;
    CH_OUT = layer.channelsOut;
    WIDTH_IN = layer.width ;
    HEIGHT_IN = layer.height;
    WIDTH_OUT = (layer.stride == 2) ? (layer.width >> 1) : layer.width;
    HEIGHT_OUT = (layer.stride == 2) ? (layer.height >> 1) : layer.height;
    // case 0, stride ==1, LOOP_CNT = layer.height  ;
    // case 1, stride ==2, LOOP_CNT = layer.height-1;
    LOOP_CNT  = layer.height - layer.stride + 1; 
    POOLING_TYPE = layer.globalPooling; 
    KERNEL = layer.kernel;
    STRIDE = layer.stride;
    PAD_SIZE = layer.padSize;
    RELU = layer.relu;
    IS_FIRST_SPLIT_LAYER = layer.isFirstSplitLayer;
    IS_SECOND_SPLIT_LAYER = layer.isSecondSplitLayer;
    BATCH_NORM = layer.batchNorm;
    SUB_LAYER_NUM = layer.sublayerNum; 
    SUB_LAYER_SEQ = layer.sublayerSeq;
    if(layer.sublayerNum == 1){
        // no split
        SUB_LAYER_FLAG = 0;
    }else { 
        if(layer.sublayerSeq == 0){
        // first split
            SUB_LAYER_FLAG = 1;
        }else if(layer.sublayerSeq == (layer.sublayerNum-1)){
        // end split
            SUB_LAYER_FLAG = 3;
        }else {
        // middle split
            SUB_LAYER_FLAG = 2;
        }
    }
    IMG_POS_IN = layer.imgPosIn;
    IMG_POS_OUT = layer.imgPosOut;
    WEIGHTS_POS = layer.weightsPos;
    BIAS_POS = layer.biasPos; 
    SCALES_POS = layer.scalesPos;
    MEAN_POS = layer.meanPos;
    VARIANCE_POS = layer.variancePos;
    NEXT_CH_IN = nextWeightLayer.channelsIn;
    NEXT_CH_OUT = nextWeightLayer.channelsOut;
    NEXT_KERNEL = nextWeightLayer.kernel; // kernel sizes supported: 3 or 1
    NEXT_NUM_WEIGHTS = nextWeightLayer.weightsNum;
    NEXT_LAYER_WEIGHTS_OFFSET = nextWeightLayer.memAddrWeights/sizeof(wemem_t);
    NEXT_LAYER_NUM = nextWeightLayer.layerNum; 
}
//================================================================
//= dataFlowProcess
//================================================================
void dataFlowProcess(
        wemem_t *WEIGHTS_SHARED_DRAM,
        iemem_t *READ_SHARED_DRAM,
        iemem_t *WRITE_SHARED_DRAM
		){
#pragma HLS inline off
#pragma HLS DATAFLOW


    hls_px5strm_t hlsPx5Strm    ;
    hls_out5strm_t hlsOut5Strm	;
#pragma HLS STREAM variable = hlsPx5Strm depth = 512 dim = 1
#pragma HLS STREAM variable = hlsOut5Strm depth = 512 dim = 1

    loadImageFromDRAM(
       READ_SHARED_DRAM  ,
       hlsPx5Strm
       );
   computeElement(
       hlsPx5Strm       ,
       hlsOut5Strm      ,
       WBRAM_K5A
       );
   writeBackToDRAM(
   	hlsOut5Strm      ,
       WRITE_SHARED_DRAM,
       BBRAM_K5A
       );
}
//================================================================
//= fileOpen and fileClose
//================================================================
#ifndef __SYNTHESIS__
    FILE *outFile;
    void fileOpen(layer_t &layer) {
        char fileName[50];
        char layerName[10];
        int i = 0;
        while (char c = layer.name[i]){
            layerName[i++] = (c == '/' | c == ' ') ? '_' : c;
        }
        layerName[i] = '\0';
        sprintf(fileName, "fpgaCnnProcessing_%s.bin", layerName);
        outFile = fopen(fileName, "w+");
    }
    void fileClose(){ 
        fclose(outFile); 
    }
#endif
//================================================================
//= loadWeightsFromDRAM
//================================================================
void loadWeightsFromDRAM(
        wemem_t *SHARED_DRAM,
        apfix32_weights_t WBRAM_K5[BLOCK_SIZE][PE_K5_CO_PORT][PE_CI][SQR_K5_SIZE],
        apfix32_weights_t BBRAM_K5[BLOCK_SIZE][BIAS_NUM][WEMEM_1ADOTS_DIV8][PE_K5_CO_PORT]
	){
#pragma HLS inline off

#pragma HLS ARRAY_PARTITION variable = WBRAM_K5A complete dim = 2 //co
#pragma HLS ARRAY_PARTITION variable = WBRAM_K5A complete dim = 3 //ci
#pragma HLS ARRAY_PARTITION variable = WBRAM_K5A complete dim = 4   //k
#pragma HLS RESOURCE variable = WBRAM_K5A core = RAM_S2P_BRAM latency = 3

#pragma HLS ARRAY_PARTITION variable = BBRAM_K5A complete dim = 2 //biasNum
#pragma HLS ARRAY_PARTITION variable = BBRAM_K5A complete dim = 3 //coDiv8
#pragma HLS ARRAY_PARTITION variable = BBRAM_K5A complete dim = 4 //PE_K5_CO_PORT
#pragma HLS RESOURCE variable = BBRAM_K5A core = RAM_S2P_BRAM latency = 3
    //Setup
    kernel_t nextKernel = NEXT_KERNEL;
    channels_t nextChIn = NEXT_CH_IN;
    channels_t nextChOut = NEXT_CH_OUT;
    channels_t nextChOutDiv16 = CEIL_DIV(nextChOut,WEMEM_1ADOTS); 
    channels_t nextChOutAlignment = nextChOutDiv16 * WEMEM_1ADOTS;
    wemem_t WBuff[MAX_WBUFF_SIZE];
    filter_t filterSize = nextKernel * nextKernel; 
    offset_t filterMulChOut = filterSize * nextChOut; 
    //Load bias coefficients
    memaddr_t addrBaseOffset = (WEIGHTS_OFFSET+LAYER_BIAS_OFFSET) ;
    memaddr_t dramAddr = 0;
    memcpy(WBuff,&SHARED_DRAM[addrBaseOffset],SIZEOF_WEMEM * BIAS_NUM * nextChOutDiv16);
    L_LOAD_BIAS_FROM_DRAM:
    for (bias_t biasId = 0; biasId < BIAS_NUM; biasId++){
        for(channels_t coDiv16 = 0; coDiv16 < nextChOutDiv16 ; coDiv16++){
        #pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 2 MAX = 16
        #pragma HLS PIPELINE II = 1    
            apfix32_weights_t apfix32Bias[WEMEM_1ADOTS_DIV4];
			#pragma HLS ARRAY_PARTITION variable = apfix32Bias complete dim = 0
            wemem_t biasData = WBuff[dramAddr++];
            ExtMemToApFixSync<32,wemem_t,apfix32_weights_t,4>(biasData,apfix32Bias);
            for(emem_1adots_t m = 0; m < WEMEM_1ADOTS_DIV4; m++){
	        #pragma HLS unroll
            	BBRAM_K5[coDiv16][biasId][m/PE_K5_CO_PORT][m%PE_K5_CO_PORT] = apfix32Bias[m];
//            	#ifndef __SYNTHESIS__
//            	    fprintf(outFile,"bias %16x \n",(int)apfix32Bias[m]);
//            	#endif
            }
        }
    }
    //Load weights coefficients
    addrBaseOffset = (WEIGHTS_OFFSET+LAYER_WEIGHTS_OFFSET) ;
    L_LOAD_WEIGHTS_FROM_DRAM:
    for (channels_t ci =0; ci < nextChIn; ci++ ){
    #pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 2 MAX = 16
    	dramAddr = 0;
        memaddr_t addrOffset = addrBaseOffset + ci * filterMulChOut * DOT_BYTES / SIZEOF_WEMEM ;
        memcpy(WBuff,&SHARED_DRAM[addrOffset],filterMulChOut * DOT_BYTES);
        L_LOAD_FILTERS_FROM_DRAM:
        for (filter_t k = 0; k < filterSize; k++){
        #pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 9 MAX = 25 
            L_LOAD_CHOUTDIV_FROM_DRAM:
            for(channels_t coDiv8 = 0; coDiv8 < nextChOutDiv16*2; coDiv8++){
                #pragma HLS PIPELINE II = 1
                //First 		: Traverse co , from 0 to nextChOut
                //Second  	: Traverse k  , from 0 to 24 (nextKernel=5)
                //Third		: Traverse ci , from 0 to nextChIn
                apfix32_weights_t apfix32Weights[WEMEM_1ADOTS_DIV4];
                #pragma HLS ARRAY_PARTITION variable = apfix32Weights complete dim = 0
                if(!(coDiv8%2)){
                    wemem_t weights = WBuff[dramAddr++];
                    ExtMemToApFixSync<32,wemem_t,apfix32_weights_t,4>(weights,apfix32Weights);
                }
                //Calucate the Address
                channels_t ciId = ci % PE_K5_CI;
                block_size_t blkSizeId = (coDiv8 * 8 + (ci/PE_K5_CI) * nextChOut )/(PE_K5_CI * PE_K5_CO_PORT);
                emem_1adots_t baseAddr = (coDiv8 % PE_K5_CO_PORT ) * PE_K5_CO_PORT;
                //Write weights to WBRAM_K5
                for(emem_1adots_t m = 0; m < PE_K5_CO_PORT; m++){
                #pragma HLS unroll
                	WBRAM_K5[blkSizeId][m][ciId][k] = apfix32Weights[baseAddr+m];
//					#ifndef __SYNTHESIS__
//						fprintf(outFile,"wbram %16x \n",(int)apfix32Weights[baseAddr+m]);
//					#endif
                }
            }
        }
    }
}
//================================================================
//= loadImageFromDRAM
//================================================================
void loadImageFromDRAM(
    iemem_t *READ_SHARED_DRAM   ,
    hls_px5strm_t &hlsPx5Strm
    ){
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable = IBRAM_K5 complete dim = 2 //y
#pragma HLS ARRAY_PARTITION variable = IBRAM_K5 complete dim = 3 //chin
#pragma HLS RESOURCE variable = IBRAM_K5 core = RAM_S2P_BRAM latency = 3
    printf(" LoadImageFromDram Function data path                                 \n");        
    printf(" READ_SHARED_DRAM --> IBuff --> IBRAM_K5 --> pixChInColsK5            \n");
    printf(" --> winPixK5 and bufferPixK5 --> px5StrmValue -->hlsPx5Strm          \n");
#ifndef __SYNTHESIS__
    data_t maxData = 0;
    data_t minData = 127;
#endif
    //IBuff
	iemem_t IBuff[MAX_WIDTH_X_CHIN_DIV16];
    offset_t lineWidthDiv16 = CEIL_DIV(DDR_CH_IN,PE_CI * BRAM_1ADOTS) * WIDTH_IN ;
	data_t pixChInColsK5[MAX_NUM_SPLIT_CHOUT/PE_CI][PE_CI][K5_SIZE];
	#pragma HLS ARRAY_PARTITION variable = pixChInColsK5 complete dim = 2
	#pragma HLS ARRAY_PARTITION variable = pixChInColsK5 complete dim = 3 
	win_pix_k5_t winPixK5[MAX_NUM_SPLIT_CHOUT/BRAM_1ADOTS][BRAM_1ADOTS];
	#pragma HLS ARRAY_PARTITION variable = winPixK5 complete dim = 2
	data_t bufferPixK5[BRAM_1ADOTS][K5_SIZE][K5_SIZE];
	#pragma HLS ARRAY_PARTITION variable = bufferPixK5 complete dim = 0
    channels_t chInPEDiv4 = CEIL_DIV(CH_IN,BRAM_1ADOTS);
    L_LOAD_IMAGES_FROM_DRAM_Y:
    for(dimension_t y = 0; y < LOOP_CNT + K5_SIZE - PAD_SIZE ; y++  ){
    #pragma HLS LOOP_TRIPCOUNT min = 12 max = 36 avg = 20
        //Load One line width images to IBuff
        if( y < HEIGHT_IN ){
            memaddr_t dramAddr = IMAGE_OFFSET + LAYER_IN_IMAGE_OFFSET + y * lineWidthDiv16;
            memcpy(IBuff,&READ_SHARED_DRAM[dramAddr],sizeof(iemem_t) * lineWidthDiv16);
        }
        //reset the bufAddr;
        offset_t bufAddr = 0;
        L_LOAD_IMAGES_FROM_DRAM_X:
        for(dimension_t x = -(PAD_SIZE * 2); x < WIDTH_IN; x++){
        #pragma HLS LOOP_TRIPCOUNT min = 12 max = 36 avg = 20
            L_LOAD_IMAGES_FROM_DRAM_CIDIV4:
            for(dimension_t ciDiv4 = 0; ciDiv4 < chInPEDiv4 ; ciDiv4++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 16 avg = 8 
            #pragma HLS pipeline II = 1
                //Write to IBRAM_K5
                if( (y < HEIGHT_IN) && (x >= 0) && ( (ciDiv4 % PE_CI )==0) ){
                    iemem_t vDatas = IBuff[bufAddr++];
                    #ifndef __SYNTHESIS__
						data_t vData[IEMEM_1ADOTS];
						ExtMemToApFixSync<DOT_BITS, iemem_t,data_t,IEMEM_1ADOTS >(vDatas,vData);
						for(offset_t vDataCnt = 0; vDataCnt < IEMEM_1ADOTS; vDataCnt++){
							if(maxData < vData[vDataCnt]){
								maxData  = vData[vDataCnt];
							}
							if(minData > vData[vDataCnt]){
								minData  = vData[vDataCnt];
							}
						}
//						fprintf(outFile,"%4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d \n",
//								 (short)vData[15],(short)vData[14],(short)vData[13],(short)vData[12],
//								 (short)vData[11],(short)vData[10],(short)vData[9],(short)vData[8],
//								 (short)vData[7],(short)vData[6],(short)vData[5],(short)vData[4],
//								 (short)vData[3],(short)vData[2],(short)vData[1],(short)vData[0]
//								 );
                    #endif
                    apfix32_image_t vIBRAM[BRAM_1ADOTS];
	                #pragma HLS ARRAY_PARTITION variable = vIBRAM complete dim = 0
                    ExtMemToApFixSync<BRAM_WIDTH, iemem_t,apfix32_image_t,BRAM_1ADOTS >(vDatas,vIBRAM);
                    offset_t yLineId = y % NUM_IMG_CACHE_LINES;
                    offset_t xLineId = ( (x * DDR_CH_IN + ciDiv4 * BRAM_1ADOTS)/(BRAM_1ADOTS * PE_CI) ) % MAX_WIDTH_X_CHIN_DIV16;
                    L_WRITE_TO_IBRAM_K5:
                    for(offset_t i = 0; i < PE_CI ; i++ ){
                    #pragma HLS unroll
                        IBRAM_K5[xLineId][yLineId][i] = vIBRAM[i];
                    }
                }
                bool skipPixel = (STRIDE == 2) & ((x % 2) | (y %2));
                // Center methord, Write to pixChInColsK5
                if( y >= (K5_SIZE-PAD_SIZE) ){
                    px5strm_t px5StrmValue;
                    dimension_t pixCenterX = x + PAD_SIZE; 
                    offset_t iAddr = (pixCenterX < 0)?offset_t(0):offset_t(((pixCenterX * DDR_CH_IN +ciDiv4 * BRAM_1ADOTS)/(PE_CI * BRAM_1ADOTS))%MAX_WIDTH_X_CHIN_DIV16);
                    //Write to pixChInColsK5, 4x5=20dots, one clock
                    L_LOOP_WRITE_TO_PIXCHINCOLSK5:
                    for(filter_t hKernel = 0; hKernel < K5_SIZE; hKernel++){
                    #pragma HLS UNROLL
                        // j = 0,1,2,3,4
                        // y - K5_SIZE = -2 ....
                        // y + j - K5_SIZE = -2,-1,0,1,2
                        dimension_t pixCenterY = y + hKernel - K5_SIZE;
                        bool isPadPixel = pixCenterX < 0 | pixCenterX >= WIDTH_IN | pixCenterY < 0 | pixCenterY >= HEIGHT_IN;
                        offset_t offsetY = isPadPixel ? offset_t(0) : offset_t(pixCenterY % NUM_IMG_CACHE_LINES);
                        offset_t offsetCiDiv16 = ciDiv4 % PE_CI;
                        data_t pixTmp[BRAM_1ADOTS];
	                    #pragma HLS ARRAY_PARTITION variable = pixTmp complete dim = 0
                        ExtMemToApFixSync<DOT_BITS, apfix32_image_t,data_t,BRAM_1ADOTS >(IBRAM_K5[iAddr][offsetY][offsetCiDiv16],pixTmp);
                        for(offset_t ciDots =0; ciDots < BRAM_1ADOTS; ciDots++){
                        #pragma HLS UNROLL
                            pixChInColsK5[ciDiv4][ciDots][hKernel] = isPadPixel ? data_t(0) : pixTmp[ciDots];
                        }
                    }
                    //pixChinColsK5 --> bufferPixK5 and winPixK5
                    // One Clock , insert 4x5 =20 dots from pixChInColsK5
                    // One Clock , generate 4x5x5 = 100 dots by sliding window for next computing 
                    for(offset_t ciDots = 0; ciDots < BRAM_1ADOTS; ciDots++ ){
                    #pragma HLS UNROLL
                        if( 1 == chInPEDiv4 ){
                        // winPixK5 No need 
                            // One clock ,generate 25 x 4 = 100 dots for computing
                            for(filter_t hKernel = 0; hKernel < K5_SIZE; hKernel++){
                            #pragma HLS UNROLL
                                for(filter_t wKernel = 0; wKernel < K5_SIZE-1; wKernel++){
                                #pragma HLS UNROLL
                                    //// left shift
                                    bufferPixK5[ciDots][hKernel][wKernel] = bufferPixK5[ciDots][hKernel][wKernel+1];
                                }
                                bufferPixK5[ciDots][hKernel][K5_SIZE-1] = pixChInColsK5[ciDiv4][ciDots][hKernel]; 
                            }
                        }else{
                            winPixK5[ciDiv4][ciDots].shift_pixels_left();
                            winPixK5[ciDiv4][ciDots].insert_right_col(pixChInColsK5[ciDiv4][ciDots]);
                        }
                    }
                    //bufferPixK5 and winPixK5 --> px5StrmValue;
                    if( x >= 0 ){
                        for(offset_t ciDots = 0; ciDots < BRAM_1ADOTS; ciDots++ ){
                        #pragma HLS UNROLL
                            for(filter_t hKernel = 0; hKernel <  K5_SIZE; hKernel++){
                            #pragma HLS UNROLL
                                for(filter_t wKernel = 0; wKernel <  K5_SIZE; wKernel++){
                                #pragma HLS UNROLL
                                    offset_t index = ciDots * K5_SIZE * K5_SIZE + hKernel * K5_SIZE + wKernel;
                                    data_t bufPix;
                                    if( 1 == chInPEDiv4 ){
                                        bufPix = bufferPixK5[ciDots][hKernel][wKernel];
                                    }else{
                                        bufPix = winPixK5[ciDiv4][ciDots].val[hKernel][wKernel];
                                    }
                                    hls::AXISetBitFields<PX5STRM_WIDTH, data_t>(px5StrmValue, index * DOT_BITS , DOT_BITS, bufPix);
                                }
                            }
                        }
                        //px5StrmValue --> hlsPx5Strm;
                        if(!skipPixel){
                            hlsPx5Strm << px5StrmValue;
                        }
                    }//End if x >=0
                }//End if y >= (K5_SIZE-PAD_SIZE)
            }//End ciDiv4
        }//End X
    }//End Y
#ifndef __SYNTHESIS__
    printf("Image max data: %4d, min data:%4d \n",short(maxData),short(minData));
#endif

}//End loadImageFromDRAM
//================================================================
//= computeElement 
//================================================================
void computeElement(
        hls_px5strm_t &hlsPx5Strm                                                   ,
        hls_out5strm_t &hlsOut5Strm                                                 ,
        apfix32_weights_t WBRAM_K5[BLOCK_SIZE][PE_K5_CO_PORT][PE_CI][SQR_K5_SIZE]
        ){
#pragma HLS inline off
    // write here by cqiu
    result_t OUTBRAM[MAX_NUM_SPLIT_CHOUT/PE_K5_CO][PE_K5_CO];
    #pragma HLS ARRAY_PARTITION variable = OUTBRAM complete dim = 2
    #pragma DEPENDENCE variable = OUTBRAM inter flase
    result_t OUTBRAM_BUF0[PE_K5_CO];
    #pragma HLS ARRAY_PARTITION variable = OUTBRAM_BUF0 complete dim = 0 
    px5strm_t pixel5Value;
    data_t pixels5[BRAM_1ADOTS][SQR_K5_SIZE];
    #pragma HLS ARRAY_PARTITION variable = pixels5 complete dim = 0 
    offset_t ciOffset;
    channels_t chInPEDiv4 = CEIL_DIV(CH_IN,PE_CI);
    channels_t chOutPEDiv8 = CEIL_DIV(CH_OUT,PE_K5_CO);

    L_COMPUTE_Y:
    for(dimension_t y = 0; y < LOOP_CNT; y++){
    #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16 
        L_COMPUTE_X:
        for(dimension_t x = 0; x < WIDTH_IN; x++){
        #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16 
            L_COMPUTE_CIDIV4:
            for(channels_t ciDiv4 = 0; ciDiv4 < chInPEDiv4 ; ciDiv4++){
            #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16 
                L_PARALLEL_CODIV8:
                for(channels_t coDiv8 = 0; coDiv8 < chOutPEDiv8; coDiv8++){
                #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16 
                #pragma HLS pipeline II = 1
                    bool skipPixel = (STRIDE == 2) & (x%2 | y%2);
                    if(!skipPixel){
                        //Load pixels5 
                        if(coDiv8 == 0){
                            hlsPx5Strm >> pixel5Value;                                                 
                            //Load pixels5 
                            L_LOAD_PIXEL5_CI_PE:
                            for(channels_t ciDot = 0; ciDot < BRAM_1ADOTS; ciDot++){
                            #pragma HLS unroll
                                L_LOAD_PIXEL5_K25_PE:
                                for(channels_t k = 0; k < SQR_K5_SIZE; k++){
                                #pragma HLS unroll
                                    offset_t index = ciDot * SQR_K5_SIZE + k;
                                    hls::AXIGetBitFields<PX5STRM_WIDTH,data_t>(pixel5Value,index * DOT_BITS, DOT_BITS, pixels5[ciDot][k]);
                                }//End k
//#ifndef __SYNTHESIS__
//  if(CH_IN ==3){
//	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//			  (int)pixels5[ciDot][0], (int)pixels5[ciDot][1], (int)pixels5[ciDot][2], (int)pixels5[ciDot][3], (int)pixels5[ciDot][4]);
//	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//			  (int)pixels5[ciDot][5], (int)pixels5[ciDot][6], (int)pixels5[ciDot][7], (int)pixels5[ciDot][8], (int)pixels5[ciDot][9]);
//	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//			  (int)pixels5[ciDot][10], (int)pixels5[ciDot][11], (int)pixels5[ciDot][12], (int)pixels5[ciDot][13], (int)pixels5[ciDot][14]);
//	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//			  (int)pixels5[ciDot][15], (int)pixels5[ciDot][16], (int)pixels5[ciDot][17], (int)pixels5[ciDot][18], (int)pixels5[ciDot][19]);
//	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//			  (int)pixels5[ciDot][20], (int)pixels5[ciDot][21], (int)pixels5[ciDot][22], (int)pixels5[ciDot][23], (int)pixels5[ciDot][24]);
//  }
//#endif
                            }//End ciDot
//                            ciOffset = ciDiv4 * CEIL_DIV(CH_OUT,BRAM_1ADOTS) * BRAM_1ADOTS;
                        }//End coDiv8 ==0
                        //Load weights5
                        data_t weights5[PE_K5_CO][PE_CI][SQR_K5_SIZE];
                        #pragma HLS ARRAY_PARTITION variable = weights5 complete dim = 0 
                        getK5Weights( coDiv8 * PE_K5_CO ,ciDiv4 * PE_CI, WBRAM_K5,weights5);
                        //Compute parallel
                        L_COMPUTE_CO_MUL:
                        for(channels_t coMul = 0; coMul < PE_K5_CO/2; coMul++ ){
                        #pragma HLS UNROLL
                            result_t result[2][PE_CI];
							#pragma HLS ARRAY_PARTITION variable = result complete dim = 0
                            L_COMPUTE_CI_MUL:
                            for(channels_t ciMul = 0; ciMul < PE_CI; ciMul++){
                            #pragma HLS UNROLL
//                            	macc2dK5(pixels5[ciMul],weights5[2*coMul][ciMul],result[0][ciMul]);
//                            	macc2dK5(pixels5[ciMul],weights5[2*coMul+1][ciMul],result[1][ciMul]);
                            	macc2dK5_int8(pixels5[ciMul],weights5[2*coMul][ciMul],weights5[2*coMul+1][ciMul],result[0][ciMul],result[1][ciMul] );
                            }
                            result_t resultSum[2];
							#pragma HLS ARRAY_PARTITION variable = resultSum complete dim = 0
                            L_COMPUTE_CO_MUL_DOUBLE:
                            for(channels_t coDouble = 0; coDouble < 2; coDouble++ ){
                            #pragma HLS UNROLL
								if(CH_IN == 3){
									resultSum[coDouble] = result[coDouble][0] + result[coDouble][1] + result[coDouble][2];
								}else{
									resultSum[coDouble] = result[coDouble][0] + result[coDouble][1] + result[coDouble][2] + result[coDouble][3];
								}
								if(CEIL_DIV(CH_OUT,PE_K5_CO) == 1){
									if(ciDiv4 == 0){
										OUTBRAM_BUF0[2*coMul+coDouble] = resultSum[coDouble];
									}else{
										OUTBRAM_BUF0[2*coMul+coDouble] = OUTBRAM_BUF0[2*coMul+coDouble] + resultSum[coDouble];
									}
								}else{
									if(ciDiv4 == 0){
										OUTBRAM[coDiv8][2*coMul+coDouble] = resultSum[coDouble];
									}else{
										OUTBRAM[coDiv8][2*coMul+coDouble] = OUTBRAM[coDiv8][2*coMul+coDouble] + resultSum[coDouble];
									}
								}
                            }
                        }
                        //Sent Result 
                        if(ciDiv4 == chInPEDiv4-1 ){
                            out5strm_t out5StrmValue;
                            result_t outData[PE_K5_CO];
                            #pragma HLS ARRAY_PARTITION variable = outData complete dim = 0 
                            //Transmit the result to outData;
                            L_COMPUTE_POST:
                            for(channels_t coPost = 0 ; coPost < PE_K5_CO; coPost++){
                            #pragma HLS UNROLL
                                if(CEIL_DIV(CH_OUT,PE_K5_CO) == 1){
                                    outData[coPost] = OUTBRAM_BUF0[coPost];
                                }else{
                                    outData[coPost] = OUTBRAM[coDiv8][coPost];
                                }
                            }
                            L_COMPUTE_TRANSMIT_TO_STRM:
                            for(emem_1adots_t ememCnt = 0; ememCnt < PE_K5_CO; ememCnt++){
	                        #pragma HLS unroll

                            	if(outData[ememCnt] >= (1 << (2 * DOT_BITS-1)) ){
                            		outData[ememCnt] = (1 << (2 * DOT_BITS -1)) - 1 ;
                            	}

                            	if(outData[ememCnt] < (-(1 << (2 * DOT_BITS-1))) ){
                            		outData[ememCnt] = -(1 << (2 * DOT_BITS -1));
                            	}


                                hls::AXISetBitFields(out5StrmValue,ememCnt * DOT_BITS * 2, DOT_BITS *2 , data_result_t(outData[ememCnt]));
								#ifndef __SYNTHESIS__
//									fprintf(outFile,"y=%4d,x=%4d,coDiv8=%4d,outdata=%8f\n",
//											int(y),int(x),int(coDiv8*PE_K5_CO+ememCnt),float(outData[ememCnt]/(512*8.0f)));
									fprintf(outFile,"%8f\n",(floor(float(outData[ememCnt]/(512*8.0f))*8))/8.0f);
								#endif
                            }
                            hlsOut5Strm << out5StrmValue;
                        }// End Compute
                    }//End skipPixel
                }//End coDiv8
            }//End ciDiv4
        }//End x
    }//End y
}
//================================================================
//= getK5Weights
//================================================================
void getK5Weights(
        const channels_t coIndex                                                 ,
        const channels_t ciIndex                                                 ,
        apfix32_weights_t WBRAM[BLOCK_SIZE][PE_K5_CO_PORT][PE_CI][SQR_K5_SIZE]   ,   
        data_t weights5[PE_K5_CO][PE_CI][SQR_K5_SIZE]                         
        ){
    #pragma HLS FUNCTION_INSTANTIATE variable = coIndex
    #pragma HLS FUNCTION_INSTANTIATE variable = ciIndex
    #pragma HLS inline 
    #pragma HLS pipeline 
    //calculate address
    //ci0co0,ci0co1,ci0co2,ci0co3,ci1co0,ci1co2,ci1co3.......
    offset_t coAddrOffsetCiDiv4 = (ciIndex/PE_CI) * CEIL_DIV(CH_OUT,PE_K5_CO)*PE_K5_CO;
    block_size_t blkAddr = ( (coIndex + coAddrOffsetCiDiv4)/(BRAM_1ADOTS * PE_K5_CO_PORT) ) % BLOCK_SIZE;
    L_GETWEIGHTS_CODIV4:
    for(channels_t ci = 0; ci < PE_CI; ci++){
        L_GETWEIGHTS_KERNEL:
        for(filter_t k = 0; k < SQR_K5_SIZE; k++){
            L_GETWEIGHTS_coDiv4:
            for(channels_t coDiv4 = 0; coDiv4 < PE_K5_CO_PORT; coDiv4++){
                data_t weightsValue[BRAM_1ADOTS];
                ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>(WBRAM[blkAddr][coDiv4][ci][k],weightsValue);
                weights5[coDiv4 * BRAM_1ADOTS + 0][ci][k] = weightsValue[0];
                weights5[coDiv4 * BRAM_1ADOTS + 1][ci][k] = weightsValue[1];
                weights5[coDiv4 * BRAM_1ADOTS + 2][ci][k] = weightsValue[2];
                weights5[coDiv4 * BRAM_1ADOTS + 3][ci][k] = weightsValue[3];
            }
        }
    }
}
//================================================================
//= macc2dK5_int8
//================================================================
void macc2dK5Int8Tmp(
        const data_t pixels5[SQR_K5_SIZE]   ,
        const data_t weights5Low[SQR_K5_SIZE]  , const data_t weights5High[SQR_K5_SIZE]  ,
         result_t &resultLow, result_t &resultHigh) {
#pragma HLS inline
	macc2dK5(pixels5,weights5Low,resultLow);
	macc2dK5(pixels5,weights5High,resultHigh);
}

//================================================================
//= macc2dK5_int8
//================================================================
void macc2dK5_int8(
        const data_t pixels5[SQR_K5_SIZE]   ,
        const data_t weights5Low[SQR_K5_SIZE]  , const data_t weights5High[SQR_K5_SIZE]  ,
            result_t &resultLow, result_t &resultHigh ) {
#pragma HLS inline

#define  HIGH_SHIFT_BITS 18
    apfix45_dsp_out_t mulResult[SQR_K5_SIZE];
    apfix45_dsp_out_t accumulator[ACCU_GROUP] = { 0, 0, 0, 0 };
    #pragma HLS ARRAY_PARTITION variable = mulResult complete dim = 0

    apfix27_weight_t weights5_comb[SQR_K5_SIZE];

    result_t accuResultLow[ACCU_GROUP];
    result_t accuResultHigh[ACCU_GROUP];

    int pixels5_int[SQR_K5_SIZE];

    #ifndef __SYNTHESIS__
      if(CH_IN ==3){
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//    			  (int)pixels5[0], (int)pixels5[1], (int)pixels5[2], (int)pixels5[3], (int)pixels5[4]);
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//    			  (int)pixels5[5], (int)pixels5[6], (int)pixels5[7], (int)pixels5[8], (int)pixels5[9]);
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//    			  (int)pixels5[10], (int)pixels5[11], (int)pixels5[12], (int)pixels5[13], (int)pixels5[14]);
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//    			  (int)pixels5[15], (int)pixels5[16], (int)pixels5[17], (int)pixels5[18], (int)pixels5[19]);
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//    			  (int)pixels5[20], (int)pixels5[21], (int)pixels5[22], (int)pixels5[23], (int)pixels5[24]);

//    	  fprintf(outFile,"weights: [ %4d %4d %4d %4d %4d ] \n",
//    			  (int)weights5[0], (int)weights5[1], (int)weights5[2], (int)weights5[3], (int)weights5[4]);
//    	  fprintf(outFile,"weights: [ %4d %4d %4d %4d %4d ] \n",
//    			  (int)weights5[5], (int)weights5[6], (int)weights5[7], (int)weights5[8], (int)weights5[9]);
//    	  fprintf(outFile,"weights: [ %4d %4d %4d %4d %4d ] \n",
//    			  (int)weights5[10], (int)weights5[11], (int)weights5[12], (int)weights5[13], (int)weights5[14]);
//    	  fprintf(outFile,"weights: [ %4d %4d %4d %4d %4d ] \n",
//    			  (int)weights5[15], (int)weights5[16], (int)weights5[17], (int)weights5[18], (int)weights5[19]);
//    	  fprintf(outFile,"weights: [ %4d %4d %4d %4d %4d ] \n",
//    			  (int)weights5[20], (int)weights5[21], (int)weights5[22], (int)weights5[23], (int)weights5[24]);

      }

    #endif

      L_MACC_COMB:
      for (int i = 0; i < SQR_K5_SIZE; i++) {
    #pragma HLS UNROLL
    	  weights5_comb[i] = (( apfix27_weight_t( weights5High[i] ))<<HIGH_SHIFT_BITS) +apfix27_weight_t( weights5Low[i]);
      }

    L_MACC_MULTIPLY:
    for(filter_t i = 0; i< SQR_K5_SIZE; i++){
    #pragma HLS UNROLL
        mulResult[i] = pixels5[i] * weights5_comb[i] ;
    }

     L_MACC_ACCUMULATE:
	for( accu_group_t group=0; group<ACCU_GROUP; group++ ){
#pragma HLS UNROLL
        for(int i = group*INT8_TIMES; i< ( group+1)*INT8_TIMES; i++){
        	if( i<SQR_K5_SIZE ){
        		 accumulator[group] = accumulator[group] + mulResult[i] ;
        	}
        }
	}

	L_LOW_ACCUMULATE_RSULT:
    for( int i=0; i<ACCU_GROUP; i++ ) {
    	accuResultLow[i] = ( result_t )( accumulator[i] & BIT_16_TO_0 );
        if( accumulator[i] & BIT_17  ){
        	accuResultLow[i] = accuResultLow[i] | BIT_47_TO_17;
        } else {
        	accuResultLow[i] = accuResultLow[i] & BIT_16_TO_0;
        }

        if( i==0 ) {
        	resultLow = accuResultLow[i];
        }
        else{
        	resultLow = resultLow + accuResultLow[i];
        }
    }

    L_HIGH_ACCUMULATE_RSULT:
    for( int i=0; i<ACCU_GROUP; i++ ) {
    	accuResultHigh[i] = ( result_t )(( accumulator[i] & 0x00007fffc0000 )>>HIGH_SHIFT_BITS );
        if( accumulator[i] & BIT_35  ){
        	accuResultHigh[i] = accuResultHigh[i] | BIT_47_TO_17;
        } else {
        	accuResultHigh[i] = accuResultHigh[i] & BIT_16_TO_0;
        }

        if( i==0 ){
        	resultHigh = accuResultHigh[i];
        }
        else{
        	resultHigh = resultHigh + accuResultHigh[i];
        }
    }

    for( int i=0; i<ACCU_GROUP; i++ ){
    	if( accumulator[i] & BIT_17  ){
    		resultHigh = resultHigh + 1;
    	}
    }
}           


void macc2dK5(
        const data_t pixels5[SQR_K5_SIZE]   ,
        const data_t weights5[SQR_K5_SIZE]  ,
            result_t &result ) {
#pragma HLS inline
    result_t accumulator = 0;
    data_result_t mulResult[SQR_K5_SIZE];
    #pragma HLS ARRAY_PARTITION variable = mulResult complete dim = 0

    #ifndef __SYNTHESIS__
      if(CH_IN ==3){
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//    			  (int)pixels5[0], (int)pixels5[1], (int)pixels5[2], (int)pixels5[3], (int)pixels5[4]);
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//    			  (int)pixels5[5], (int)pixels5[6], (int)pixels5[7], (int)pixels5[8], (int)pixels5[9]);
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//    			  (int)pixels5[10], (int)pixels5[11], (int)pixels5[12], (int)pixels5[13], (int)pixels5[14]);
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//    			  (int)pixels5[15], (int)pixels5[16], (int)pixels5[17], (int)pixels5[18], (int)pixels5[19]);
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
//    			  (int)pixels5[20], (int)pixels5[21], (int)pixels5[22], (int)pixels5[23], (int)pixels5[24]);

//    	  fprintf(outFile,"weights: [ %4d %4d %4d %4d %4d ] \n",
//    			  (int)weights5[0], (int)weights5[1], (int)weights5[2], (int)weights5[3], (int)weights5[4]);
//    	  fprintf(outFile,"weights: [ %4d %4d %4d %4d %4d ] \n",
//    			  (int)weights5[5], (int)weights5[6], (int)weights5[7], (int)weights5[8], (int)weights5[9]);
//    	  fprintf(outFile,"weights: [ %4d %4d %4d %4d %4d ] \n",
//    			  (int)weights5[10], (int)weights5[11], (int)weights5[12], (int)weights5[13], (int)weights5[14]);
//    	  fprintf(outFile,"weights: [ %4d %4d %4d %4d %4d ] \n",
//    			  (int)weights5[15], (int)weights5[16], (int)weights5[17], (int)weights5[18], (int)weights5[19]);
//    	  fprintf(outFile,"weights: [ %4d %4d %4d %4d %4d ] \n",
//    			  (int)weights5[20], (int)weights5[21], (int)weights5[22], (int)weights5[23], (int)weights5[24]);

      }

    #endif

    L_MACC_MULTIPLY:
    for(filter_t i = 0; i< SQR_K5_SIZE; i++){
    #pragma HLS UNROLL
        mulResult[i] = pixels5[i] * weights5[i] ;
    }
    L_MACC_ACCUMULATE:
    for(filter_t i = 0; i< SQR_K5_SIZE; i++){
    #pragma HLS UNROLL
       accumulator = accumulator + result_t(mulResult[i]);
    }
    result = accumulator ;


//    for( int i=0; i<SQR_K5_SIZE; i++ ){
//    	printf("i=%d; pixels5=%d\n",i, int(pixels5[i]) );
//    }
//
//    for( int i=0; i<SQR_K5_SIZE; i++ ){
//    	printf("i=%d; weights5=%d\n",i, int(weights5[i]) );
//    }
//
//
//   printf("result=%d\n",int(result) );



//    printf("result = %4d\n",int(result));

}

//================================================================
//= writeBackToDRAM
//================================================================
void writeBackToDRAM(
        hls_out5strm_t &hlsOut5Strm,                                    
        iemem_t *WRITE_SHARED_DRAM,
        apfix32_weights_t BBRAM_K5[BLOCK_SIZE][BIAS_NUM][WEMEM_1ADOTS_DIV8][PE_K5_CO_PORT]
    ){
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable = LBRAM complete dim = 2 // EXTMEM_ALIGNMENT
#pragma HLS RESOURCE variable = LBRAM core = RAM_T2P_BRAM latency = 3
//data_t LBRAM[MAX_WIDTH_X_CHIN_DIV16][WEMEM_1ADOTS];////512x16
data_t maxData = 0;
data_t minData = 255;
dimension_t widthOut = (STRIDE ==2) ? dimension_t(WIDTH_IN/2) : WIDTH_IN ;
bool isSplitLayer = IS_FIRST_SPLIT_LAYER | IS_SECOND_SPLIT_LAYER   ; 
num_sublayers_t subLayerMulti = (SUB_LAYER_FLAG == 0) ? num_sublayers_t(1) : SUB_LAYER_NUM   ;
offset_t xOffsetDiv16 = subLayerMulti * CEIL_DIV(CH_OUT,IEMEM_1ADOTS);
offset_t yOffsetDiv16 = xOffsetDiv16 * WIDTH_IN ;
    L_WRITEBACK_Y:
    for(dimension_t y = 0 ; y < HEIGHT_IN + 1; y++){
    #pragma HLS LOOP_TRIPCOUNT MIN = 8 AVG = 16 MAX = 32 
        L_WRITEBACK_X:
        for(dimension_t x = 0; x < WIDTH_IN; x++){
        #pragma HLS LOOP_TRIPCOUNT MIN = 8 AVG = 16 MAX = 32 
            L_WRITEBACK_CODIV16:
            for(channels_t coDiv8 = 0; coDiv8 < CEIL_DIV(CH_OUT,PE_K5_CO); coDiv8++){
            #pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 2 MAX = 4 
            #pragma HLS pipeline II = 1
                data_result_t raw[PE_K5_CO];
                result_t bias[PE_K5_CO];
                result_t mean[PE_K5_CO];
                result_t variance[PE_K5_CO];
                result_t batchData[PE_K5_CO];
                result_t batchDataRelu[PE_K5_CO];
                result_t batchDataDiv10Int[PE_K5_CO];
                data_t activeData[PE_K5_CO];
                #pragma HLS ARRAY_PARTITION variable = raw complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = variance complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = mean complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = bias complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = batchData complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = batchDataRelu complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = batchDataDiv10Int complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = activeData complete dim = 0 
                data_t preX[MAX_CHANNELS/PE_K5_CO][PE_K5_CO];
                data_t curX[MAX_CHANNELS/PE_K5_CO][PE_K5_CO];
                data_t maxX[MAX_CHANNELS/PE_K5_CO][PE_K5_CO];
                data_t tmpX[MAX_CHANNELS/PE_K5_CO][PE_K5_CO];
                #pragma HLS ARRAY_PARTITION variable = preX complete dim = 2
                #pragma HLS ARRAY_PARTITION variable = curX complete dim = 2
                #pragma HLS ARRAY_PARTITION variable = maxX complete dim = 2
                #pragma HLS ARRAY_PARTITION variable = tmpX complete dim = 2
                memaddr_t ddrAddr ;
                half_iemem_t halfIememTmp;
                iemem_t iememTmp;
                //The last row process
                if( y == HEIGHT_IN ){
                    if(POOLING_TYPE == 1){// Max Pooling Stride == 1 
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + (y-1) * yOffsetDiv16 + x * xOffsetDiv16 + coDiv8/2;  
                        ApFixToExtMemSync<DOT_BITS,half_iemem_t,data_t,PE_K5_CO>(halfIememTmp,LBRAM[offset_t(x * xOffsetDiv16 *2 + coDiv8)]);
                    }
                }else{
                //Normal rows process
                    //Load raws values 
                    out5strm_t outResult;
                    hlsOut5Strm >> outResult;
                    ExtMemToApFixSync<DOT_BITS * 2 ,out5strm_t,data_result_t,PE_K5_CO>(outResult,raw);
                    data_t apfixTmp[3][PE_K5_CO_PORT][BRAM_1ADOTS];
                    //Load bias to apfixTmp[0]
                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
                        (BBRAM_K5[coDiv8/2][0][coDiv8%2][0],apfixTmp[0][0]);
                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
                        (BBRAM_K5[coDiv8/2][0][coDiv8%2][1],apfixTmp[0][1]);
//                    //Load means to apfixTmp[1]
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_K5[coDiv8/2][2][coDiv8%2][0],apfixTmp[1][0]);
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_K5[coDiv8/2][2][coDiv8%2][1],apfixTmp[1][1]);
//                    //Load variance to apfixTmp[2]
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_K5[coDiv8/2][3][coDiv8%2][0],apfixTmp[2][0]);
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_K5[coDiv8/2][3][coDiv8%2][1],apfixTmp[2][1]);
                    for(channels_t coIndex = 0; coIndex < PE_K5_CO; coIndex++){
                    #pragma HLS UNROLL factor = PE_K5_CO
                        bias[coIndex]       = apfixTmp[0][coIndex/4][coIndex%4];
                        mean[coIndex]       = apfixTmp[1][coIndex/4][coIndex%4];
                        variance[coIndex]   = apfixTmp[2][coIndex/4][coIndex%4];
                        if(BATCH_NORM == 1){
                            batchData[coIndex] = result_t ((raw[coIndex] -(mean[coIndex] << (IMG_POS_IN + WEIGHTS_POS - MEAN_POS)))) 
                                                * variance[coIndex] ;
                            batchData[coIndex] += (bias[coIndex] << (IMG_POS_IN +  WEIGHTS_POS + SCALES_POS - BIAS_POS));
                            batchDataDiv10Int[coIndex] =   ((batchData[coIndex] >> 4) + (batchData[coIndex] >> 5) + 
                                                            (batchData[coIndex] >> 8) + (batchData[coIndex] >> 9) + 
                                                            (batchData[coIndex] >>12) + (batchData[coIndex] >> 13) + 
                                                            (batchData[coIndex] >>16) + (batchData[coIndex] >> 17) );
                            batchDataRelu[coIndex] = (batchData[coIndex] > 0) ? batchData[coIndex] : batchDataDiv10Int[coIndex] ;
                            activeData[coIndex] = data_result_t(batchDataRelu[coIndex] >> (IMG_POS_IN + WEIGHTS_POS + SCALES_POS - IMG_POS_OUT));
                        }else{

                            batchData[coIndex] = raw[coIndex] + (bias[coIndex] << (IMG_POS_IN + WEIGHTS_POS - BIAS_POS)); 
                            batchDataRelu[coIndex] = (batchData[coIndex] > 0) ? batchData[coIndex] : result_t(0) ;
//                            batchDataRelu[coIndex] =  batchData[coIndex] ;
                            activeData[coIndex] = data_result_t(batchDataRelu[coIndex] >> (IMG_POS_IN + WEIGHTS_POS - IMG_POS_OUT));
							#ifndef __SYNTHESIS__
//								fprintf(outFile,"%8f\n",float(activeData[coIndex]*1.0f/(1<<IMG_POS_OUT)));
//                            	fprintf(outFile,"%8f\n",(floor(float(raw[coIndex]/(512*8.0f))*8))/8.0f);
							#endif


                        }
                        if(x == 0){
                            preX[coDiv8][coIndex] = -MAX_SHORT;
                        } else {
                            preX[coDiv8][coIndex] = curX[coDiv8][coIndex]; 
                        }
                        if(POOLING_TYPE == 0){// Max Pooling Stride == 2
                            if( (!(y%2)) && (x%2) ) {// even row, odd column
                                maxX[coDiv8][coIndex] = max(preX[coDiv8][coIndex],activeData[coIndex]);
                                LBRAM[ x * xOffsetDiv16 * 2 + coDiv8][coIndex] = maxX[coDiv8][coIndex];
                            } else if ( (y%2) && (x%2)) {// odd row, odd column
                                tmpX[coDiv8][coIndex] = max(LBRAM[x * xOffsetDiv16 *2 + coDiv8][coIndex],preX[coDiv8][coIndex]);
                                maxX[coDiv8][coIndex] = max(tmpX[coDiv8][coIndex],activeData[coIndex]);
                            }
                        } else if(POOLING_TYPE == 1){ // Max Pooling Stride == 1 
                            if(y == 0){
                                maxX[coDiv8][coIndex] = max(preX[coDiv8][coIndex],activeData[coIndex]);
                                LBRAM[ x * xOffsetDiv16 * 2 + coDiv8][coIndex] = maxX[coDiv8][coIndex]; 
                            }else{
                                tmpX[coDiv8][coIndex] = max(activeData[coIndex],preX[coDiv8][coIndex]);
                                maxX[coDiv8][coIndex] = max(tmpX[coDiv8][coIndex],data_t(LBRAM[ x * xOffsetDiv16 *2 + coDiv8][coIndex]));
                                LBRAM[ x * xOffsetDiv16 *2 + coDiv8][coIndex] = tmpX[coDiv8][coIndex];
                            }
                        }
                        curX[coDiv8][coIndex] = activeData[coIndex]; 
                    }//End coIndex


                    if(POOLING_TYPE == 0){// Max Pooling Stride == 2 
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + (y/2) * yOffsetDiv16 / 2 + (x/2) * xOffsetDiv16 + coDiv8/2;  
                        ApFixToExtMemSync<DOT_BITS,half_iemem_t,data_t,PE_K5_CO>(halfIememTmp,maxX[coDiv8]);
                    }else if(POOLING_TYPE == 2){// No Max Pooling
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + y * yOffsetDiv16  + x * xOffsetDiv16 + coDiv8/2;  
                        ApFixToExtMemSync<DOT_BITS,half_iemem_t,data_t,PE_K5_CO>(halfIememTmp,activeData);

                    }else if(POOLING_TYPE == 1){// Max Pooling Stride == 1 
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + (y-1) * yOffsetDiv16  + x * xOffsetDiv16 + coDiv8/2;  
                        ApFixToExtMemSync<DOT_BITS,half_iemem_t,data_t,PE_K5_CO>(halfIememTmp,maxX[coDiv8]);
                    }
                }//End Normal rows
                if( (   (y != HEIGHT_IN) &&
                        (((POOLING_TYPE == 0) && (y%2) && (x%2)) || (POOLING_TYPE == 2)) )
                   || ((POOLING_TYPE == 1) && (y != 0)) ){
                    hls::AXISetBitFields(iememTmp, (coDiv8%2) * HALF_IEMEM_WIDTH, HALF_IEMEM_WIDTH, halfIememTmp);
                    if( 1 == (coDiv8%2) ){
                        WRITE_SHARED_DRAM[ddrAddr] = iememTmp;

					#ifndef __SYNTHESIS__
						data_t resultValue[IEMEM_1ADOTS];
						ExtMemToApFixSync<DOT_BITS,iemem_t,data_t,IEMEM_1ADOTS>(iememTmp,resultValue);
						for(int rCnt=0;rCnt<IEMEM_1ADOTS;rCnt++){
							if(maxData < resultValue[rCnt]){
								maxData = resultValue[rCnt];
							}
							if(minData > resultValue[rCnt]){
								minData = resultValue[rCnt];
							}
//							fprintf(outFile,"%4f\n",float(resultValue[rCnt]*1.0f/(1<<IMG_POS_OUT)));
						}
					#endif
                    }
                }
            }//End coDiv8
        }//End x
    }//End y
#ifndef __SYNTHESIS__
    printf("PostProcess maxOutData %4d,minOutData %4d \n", int(maxData), int(minData));
#endif
}//End writeBackToDRAM 

