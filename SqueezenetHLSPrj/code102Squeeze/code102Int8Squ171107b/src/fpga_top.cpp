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
//-------------------------[512]-------[1]-------[2]------------[121]
apfix32_weights_t WBRAM_KLA[BLOCK_SIZE][PE_KL_CI][PE_KL_CO_PORT][SQR_KL_SIZE];
//-------------------------[512]-------[4]-------[2]----------------[2]
apfix32_weights_t BBRAM_KLA[BLOCK_SIZE][BIAS_NUM][WEMEM_1ADOTS_DIV8][PE_KL_CO_PORT];
//----------------------[2048]----------------------[16]--------------------[1]
apfix32_image_t IBRAM_KL[MAX_WIDTH_X_CHIN_DIV4_KL][NUM_IMG_CACHE_LINES_KL][PE_KL_CI];
//----------[512]------------------------[8]
data_t LBRAM[MAX_WIDTH_X_CHIN_DIV16_KL*2][PE_KL_CO];////512x16

//-------------------------[512]-------[4]-------[16]-----------[9]
apfix32_weights_t WBRAM_KSA[BLOCK_SIZE][PE_KS_CI][PE_KS_CO_PORT][SQR_KS_SIZE];
//-------------------------[512]-------[4]-------[4]-----------------[4]
apfix32_weights_t BBRAM_KSA[BLOCK_SIZE][BIAS_NUM][PE_KS_CO_PORT_DIV4][WEMEM_1ADOTS_DIV4];
//----------------------[1024]---------------------[4]---------------------[4]
apfix32_image_t IBRAM_KS[MAX_WIDTH_X_CHIN_DIV16_KS][NUM_IMG_CACHE_LINES_KS][PE_KS_CI];
//-------------[1024]---------------------[16]
data_t LBRAM_KS[MAX_WIDTH_X_CHIN_DIV16_KS][IEMEM_1ADOTS];////1024x16


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
memaddr_t NEXT_LAYER_WEIGHTS_OFFSET ;
num_layers_t NEXT_LAYER_NUM			;
filter_t KL_REM                     ;////remainder
dimension_t KL_PAD_SIZE_EXTEND      ;
dimension_t KL_WIDTH                ;
dimension_t KL_HEIGHT               ;
dimension_t KL_WIDTH_STRIDE         ;
dimension_t KL_HEIGHT_STRIDE        ;
filter_t KS_REM                     ;////remainder
dimension_t KS_PAD_SIZE_EXTEND      ;
dimension_t KS_WIDTH                ;
dimension_t KS_HEIGHT               ;
dimension_t KS_WIDTH_STRIDE         ;
dimension_t KS_HEIGHT_STRIDE        ;

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
    loadWeightsFromDRAMKL(
    		WEIGHTS_SHARED_DRAM,
			WBRAM_KLA,
			BBRAM_KLA);
    dataFlowProcessKL(
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
    NEXT_KERNEL = nextWeightLayer.kernel; // kernel sizes supported: 11,7,5,3 or 1
    NEXT_LAYER_WEIGHTS_OFFSET = nextWeightLayer.memAddrWeights/sizeof(wemem_t);
    NEXT_LAYER_NUM = nextWeightLayer.layerNum; 

	// 11x11 deal with 7x7,PAD_SIZE=0, so padSizeExtend = 2
	KL_REM = (KL_SIZE-KERNEL)/2;////remainder
	KL_PAD_SIZE_EXTEND = PAD_SIZE + dimension_t(KL_REM);
	KL_WIDTH           = (1+(WIDTH_IN+2*KL_PAD_SIZE_EXTEND-KL_SIZE)/STRIDE);
	KL_HEIGHT          = (1+(HEIGHT_IN+2*KL_PAD_SIZE_EXTEND-KL_SIZE)/STRIDE);
	KL_WIDTH_STRIDE    = STRIDE*KL_WIDTH;
	KL_HEIGHT_STRIDE   = STRIDE*KL_HEIGHT;
	// 3x3 Kernel
	KS_REM = (KS_SIZE-KERNEL)/2;////remainder
	KS_PAD_SIZE_EXTEND = PAD_SIZE + dimension_t(KS_REM);
	KS_WIDTH           = (1+(WIDTH_IN+2*KS_PAD_SIZE_EXTEND-KS_SIZE)/STRIDE);
	KS_HEIGHT          = (1+(HEIGHT_IN+2*KS_PAD_SIZE_EXTEND-KS_SIZE)/STRIDE);
	KS_WIDTH_STRIDE    = STRIDE*KS_WIDTH;
	KS_HEIGHT_STRIDE   = STRIDE*KS_HEIGHT;

}

//================================================================
//= loadWeightsFromDRAMKS
//================================================================
void loadWeightsFromDRAMKS(
        wemem_t *SHARED_DRAM,
        apfix32_weights_t WBRAM_KS[BLOCK_SIZE][PE_KS_CI][PE_KS_CO_PORT][SQR_KS_SIZE],
		apfix32_weights_t BBRAM_KS[BLOCK_SIZE][BIAS_NUM][PE_KS_CO_PORT_DIV4][WEMEM_1ADOTS_DIV4]
	){
#pragma HLS inline off

#pragma HLS ARRAY_PARTITION variable = WBRAM_KSA complete dim = 2 //ci
#pragma HLS ARRAY_PARTITION variable = WBRAM_KSA complete dim = 3 //co
#pragma HLS ARRAY_PARTITION variable = WBRAM_KSA complete dim = 4   //k
#pragma HLS RESOURCE variable = WBRAM_KSA core = RAM_S2P_BRAM latency = 3

#pragma HLS ARRAY_PARTITION variable = BBRAM_KSA complete dim = 2 //biasNum
#pragma HLS ARRAY_PARTITION variable = BBRAM_KSA complete dim = 3 //PE_KS_CO_PORT_DIV4
#pragma HLS ARRAY_PARTITION variable = BBRAM_KSA complete dim = 4 //WEMEM_1ADOTS_DIV4
#pragma HLS RESOURCE variable = BBRAM_KSA core = RAM_S2P_BRAM latency = 3
    //Setup
    kernel_t nextKernel = NEXT_KERNEL;
    channels_t nextChIn = NEXT_CH_IN;
    channels_t nextChOut = NEXT_CH_OUT;
    channels_t nextChOutDiv16 = CEIL_DIV(nextChOut,WEMEM_1ADOTS);
    channels_t nextChOutAlignment = nextChOutDiv16 * WEMEM_1ADOTS;
    wemem_t WBuff[MAX_WBUFF_SIZE_KS];
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
            ExtMemToApFixSync<BRAM_WIDTH,wemem_t,apfix32_weights_t,WEMEM_WIDTH/BRAM_WIDTH>(biasData,apfix32Bias);
            for(emem_1adots_t m = 0; m < WEMEM_1ADOTS_DIV4; m++){
	        #pragma HLS unroll
            	BBRAM_KS[short(coDiv16/WEMEM_1ADOTS_DIV4)][biasId][short(coDiv16%WEMEM_1ADOTS_DIV4)][m] = apfix32Bias[m];
//            	#ifndef __SYNTHESIS__
//            	    fprintf(outFile,"bias %16x \n",(int)apfix32Bias[m]);
//            	#endif
            }
        }
    }
    //Load weights coefficients
    addrBaseOffset = (WEIGHTS_OFFSET+LAYER_WEIGHTS_OFFSET) ;
    filter_t fristPos = (KS_SIZE - KERNEL)*(KS_SIZE+1)/2;
    L_LOAD_WEIGHTS_FROM_DRAM:
    for (channels_t ci =0; ci < nextChIn; ci++ ){
    #pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 2 MAX = 16
    	dramAddr = 0;
        memaddr_t addrOffset = addrBaseOffset + ci * filterMulChOut * DOT_BYTES / SIZEOF_WEMEM ;
        memcpy(WBuff,&SHARED_DRAM[addrOffset],filterMulChOut * DOT_BYTES);
        L_LOAD_FILTERS_FROM_DRAM:
        for (filter_t k = 0; k < filterSize; k++){
        #pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 1 MAX = 9
            L_LOAD_CHOUTDIV_FROM_DRAM:
            for(channels_t coDiv16 = 0; coDiv16 < nextChOutDiv16; coDiv16++){
                #pragma HLS PIPELINE II = 1
                //First 	: Traverse co , from 0 to nextChOut
                //Second  	: Traverse k  , from 0 to 120 (nextKernel=11)
                //Third		: Traverse ci , from 0 to nextChIn
                apfix32_weights_t apfix32Weights[WEMEM_1ADOTS_DIV4];
                #pragma HLS ARRAY_PARTITION variable = apfix32Weights complete dim = 0

                 wemem_t weights = WBuff[dramAddr++];
                 ExtMemToApFixSync<BRAM_WIDTH,wemem_t,apfix32_weights_t,WEMEM_1ADOTS_DIV4>(weights,apfix32Weights);

//                 WBRAM_KS[BLOCK_SIZE][PE_KS_CI][PE_KS_CO_PORT][SQR_KS_SIZE],
                //Calucate the Address
                channels_t ciId = ci % PE_KS_CI;
//              filter_t sqrKTimes = FLOOR_DIV(filterSize,MAX_FILTER_SIZE);
                block_size_t blkSizeId = (coDiv16 * WEMEM_1ADOTS + (ci/PE_KS_CI) * nextChOut )/(BRAM_1ADOTS * PE_KS_CO_PORT );
                channels_t coId = coDiv16%WEMEM_1ADOTS_DIV4;
                filter_t kId = fristPos+(k%KERNEL)+(k/KERNEL)*KS_SIZE;
//				weightsL[coDiv4 * BRAM_1ADOTS + 0][ci][fristPos+(k%KERNEL)+(k/KERNEL)*KL_SIZE] = weightsValue[0];

                //Write weights to WBRAM_KL
                for(emem_1adots_t m = 0; m < WEMEM_1ADOTS_DIV4; m++){
                #pragma HLS unroll
                	WBRAM_KS[blkSizeId][ciId][coId*WEMEM_1ADOTS_DIV4+m][kId] = apfix32Weights[m];
//					#ifndef __SYNTHESIS__
//						fprintf(outFile,"wbram %16x \n",(int)apfix32Weights[baseAddr+m]);
//					#endif
                }
//#ifndef __SYNTHESIS__
//	data_t tmpW[BRAM_1ADOTS];
//	ExtMemToApFixSync<DOT_BITS, apfix32_weights_t,data_t,BRAM_1ADOTS >(WBRAM_KL[blkSizeId][ciId][1][kId],tmpW);
//	fprintf(outFile,"ci,k [%4d][%4d]=%4d \n",int(ci),int(k),(int)tmpW[2]);
//#endif
                //                if(6 == co)
                //                	printf("ci,kernel,co=[%4d],[%4d],[%4d]::%4d \n",ci,kernel,co,int(dataTmp));
            }

        }
    }
}
//================================================================
//= loadImageFromDRAM
//================================================================
void loadImageFromDRAMKS(
    iemem_t *READ_SHARED_DRAM	,
    hls_pxs_strm_t &hlsPxsStrm
    ){
#pragma HLS inline off
//	//----------------------[1024]---------------------[4]---------------------[4]
//	apfix32_image_t IBRAM_KS[MAX_WIDTH_X_CHIN_DIV16_KS][NUM_IMG_CACHE_LINES_KS][PE_KS_CI];
#pragma HLS ARRAY_PARTITION variable = IBRAM_KS complete dim = 2 //y
#pragma HLS ARRAY_PARTITION variable = IBRAM_KS complete dim = 3 //chin
#pragma HLS RESOURCE variable = IBRAM_KS core = RAM_S2P_BRAM latency = 3
    printf(" LoadImageFromDram Function data path                                 \n");
    printf(" READ_SHARED_DRAM --> IBuff --> IBRAM_KS --> pixChInColsKS            \n");
    printf(" --> winPixKS and bufferPixKS --> pxsStrmValue -->hlsPxsStrm          \n");
#ifndef __SYNTHESIS__
    data_t maxData = 0;
    data_t minData = 127;
#endif
    //IBuff
	iemem_t IBuff[MAX_WIDTH_X_CHIN_DIV16_KS];
    offset_t lineWidthDiv16 = CEIL_DIV(DDR_CH_IN,IEMEM_1ADOTS) * WIDTH_IN ;
	data_t pixChInColsKS[MAX_NUM_SPLIT_CHOUT/PE_KS_CI][PE_KS_CI][KS_SIZE];
	#pragma HLS ARRAY_PARTITION variable = pixChInColsKS complete dim = 2
	#pragma HLS ARRAY_PARTITION variable = pixChInColsKS complete dim = 3
	win_pix_ks_t winPixKS[MAX_NUM_SPLIT_CHOUT/PE_KS_CI][PE_KS_CI];
	#pragma HLS ARRAY_PARTITION variable = winPixKS complete dim = 2
	data_t bufferPixKS[PE_KS_CI][KS_SIZE][KS_SIZE];
	#pragma HLS ARRAY_PARTITION variable = bufferPixKS complete dim = 0
    channels_t chInPEDiv = CEIL_DIV(CH_IN,PE_KS_CI);
    //we use cover child algorithm.
    //Every 3x3,1x1 cnn, we all use 3x3
    //For example KS_SIZE = 3, KERNEL = 1,so remK is (3-1)/2 = 1
    // . . . . .
    // . x x x .
    // . x x x .
    // . x x x .
    // . . . . .
    // 3x3 deal with 1x1,PAD_SIZE=1, so padSizeExtend = 2
    filter_t remK = (KS_SIZE-KERNEL)/2;////remainder
    dimension_t padSizeExtend = PAD_SIZE + dimension_t(remK);
    filter_t denK = KERNEL+remK;//denominator
    dimension_t widthSize = STRIDE*(1+(WIDTH_IN+2*padSizeExtend-KS_SIZE)/STRIDE);
    dimension_t heightSize = STRIDE*(1+(HEIGHT_IN+2*padSizeExtend-KS_SIZE)/STRIDE);
    L_LOAD_IMAGES_FROM_DRAM_Y:
	// For example KS_SIZE-padSizeExtend = 9, y=0,1,2,3...8 , we should store  the y windows.
	//// here by cqiu
    for(dimension_t y = 0; y < heightSize + KS_SIZE-padSizeExtend ; y++  ){
    #pragma HLS LOOP_TRIPCOUNT min = 16 max = 269 avg = 20
        //Load One line width * chin images to IBuff
        if( y < HEIGHT_IN ){
            memaddr_t dramAddr = IMAGE_OFFSET + LAYER_IN_IMAGE_OFFSET + y * lineWidthDiv16;
            memcpy(IBuff,&READ_SHARED_DRAM[dramAddr],sizeof(iemem_t) * lineWidthDiv16);
        }
        //reset the bufAddr;
        offset_t bufAddr = 0;
        L_LOAD_IMAGES_FROM_DRAM_X:
		// For example KS_SIZE = 11, x=-10,-9,-8,-7... we should store the x windows.
        for(dimension_t x = -(KS_SIZE-1); x < WIDTH_IN; x++){
        #pragma HLS LOOP_TRIPCOUNT min = 12 max = 36 avg = 20
            L_LOAD_IMAGES_FROM_DRAM_CIDIV4:
            for(dimension_t ciDiv = 0; ciDiv < chInPEDiv ; ciDiv++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 512 avg = 8
            #pragma HLS pipeline II = 1
                //Write to IBRAM_KS
                if( (y < HEIGHT_IN) && (x >= 0) && ( ( (ciDiv*PE_KS_CI) % IEMEM_1ADOTS_DIV4 )==0) ){
                    apfix32_image_t vIBRAM[BRAM_1ADOTS];
	                #pragma HLS ARRAY_PARTITION variable = vIBRAM complete dim = 0
                    if( 0 == ((ciDiv * PE_KS_CI  ) % IEMEM_1ADOTS) ){
                        iemem_t vDatas = IBuff[bufAddr++];
                        ExtMemToApFixSync<BRAM_WIDTH, iemem_t,apfix32_image_t,IEMEM_1ADOTS_DIV4 >(vDatas,vIBRAM);
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
//									(short)vData[15],(short)vData[14],(short)vData[13],(short)vData[12],
//									(short)vData[11],(short)vData[10],(short)vData[9],(short)vData[8],
//									(short)vData[7],(short)vData[6],(short)vData[5],(short)vData[4],
//									(short)vData[3],(short)vData[2],(short)vData[1],(short)vData[0]
//								);
						#endif
                    }
                    offset_t yLineId = y % NUM_IMG_CACHE_LINES_KS;
                    //// here is wrong
                    offset_t xLineId = ( (x * DDR_CH_IN + ciDiv * PE_KS_CI)/(BRAM_1ADOTS * PE_KS_CI) ) % MAX_WIDTH_X_CHIN_DIV16_KS;
                    L_WRITE_TO_IBRAM_KS:
                    for(offset_t i = 0; i < PE_KS_CI ; i++ ){
                    #pragma HLS unroll
                        IBRAM_KS[xLineId][yLineId][i] = vIBRAM[i+((ciDiv*PE_KS_CI)%WEMEM_1ADOTS_DIV4)];
                    }
                }
                bool skipPixel = (STRIDE == 2) & ((x % 2) | ( (y-KS_SIZE+padSizeExtend) %2));
                // Valid methord, Write to pixChInColsKS
                if( y >= (KS_SIZE-padSizeExtend) ){
                    pxs_strm_t px5StrmValue;
                    // KS_SIZE-1-padSizeExtend=8
                    //        x=-10,-9,-8,-7,-6,....-2,-1,0,1, 2,....WIDTH_IN-1
                    //pixValidX= -2,-1, 0, 1, 2, 3,..6, 7,8,9,10,....WIDTH_IN+7
                    dimension_t pixValidX = x + (KS_SIZE-1-padSizeExtend);
                    offset_t iAddr = (pixValidX < 0)?offset_t(0):offset_t(((pixValidX * DDR_CH_IN +ciDiv * PE_KS_CI)/(PE_KS_CI * BRAM_1ADOTS))%MAX_WIDTH_X_CHIN_DIV16_KS);
                    //Write to pixChInColsKS, 1x11=11dots, one clock
                    L_LOOP_WRITE_TO_PIXCHINCOLSKS:
                    for(filter_t hKernel = 0; hKernel < KS_SIZE; hKernel++){
                    #pragma HLS UNROLL
                    	// y is begging from KS_SIZE-padSizeExtend,
                        // y - KS_SIZE = KS_SIZE-padSizeExtend-KS_SIZE = -2,
                    	// hKernel = 0,1,2,3,4,5,6,7,8,9,10
                        // y + hKernel - KS_SIZE = -2,-1,0,1,2,3,4,5,6,7,8
                        dimension_t pixValidY = y + hKernel - KS_SIZE;
                        bool isPadPixel = pixValidX < 0 | pixValidX >= WIDTH_IN | pixValidY < 0 | pixValidY >= HEIGHT_IN;
                        offset_t offsetY = isPadPixel ? offset_t(0) : offset_t(pixValidY % NUM_IMG_CACHE_LINES_KS);
                        offset_t offsetCiDiv = ciDiv % PE_KS_CI;
                        data_t pixTmp[BRAM_1ADOTS];
	                    #pragma HLS ARRAY_PARTITION variable = pixTmp complete dim = 0
//                        if( (0 == ((ciDiv * PE_KS_CI) % BRAM_1ADOTS)) && (!isPadPixel) ){
                        	ExtMemToApFixSync<DOT_BITS, apfix32_image_t,data_t,BRAM_1ADOTS >(IBRAM_KS[iAddr][offsetY][offsetCiDiv],pixTmp);
//                        	fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[0]));
//                        	fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[1]));
//                        	fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[2]));
//                        	fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[3]));
//                        }
                        for(offset_t ciDots =0; ciDots < PE_KS_CI; ciDots++){
                        #pragma HLS UNROLL
//                            pixChInColsKS[ciDiv][ciDots][hKernel] = isPadPixel ? data_t(0) : pixTmp[ciDots+((ciDiv*PE_KS_CI)%BRAM_1ADOTS)];
                        	pixChInColsKS[ciDiv][ciDots][hKernel] = isPadPixel ? data_t(0) : pixTmp[ciDiv];

//							fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[0]));
//							fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[1]));
//							fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[2]));
//							fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[3]));

//                        	fprintf(outFile,"fuck[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixChInColsKS[ciDiv][ciDots][hKernel]) );
//                          fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(pixValidY),int(pixValidX),int(ciDiv),int(hKernel),int(pixChInColsKS[ciDiv][ciDots][hKernel]));
                        }
                    }
                    //pixChinColsKS --> bufferPixKS and winPixKS
                    // One Clock , insert 4x5 =20 dots from pixChInColsKS
                    // One Clock , generate 4x5x5 = 100 dots by sliding window for next computing
                    for(offset_t ciDots = 0; ciDots < PE_KS_CI; ciDots++ ){
                    #pragma HLS UNROLL
                        if( 1 == chInPEDiv ){
                        // winPixKS No need
                            // One clock ,generate 121 x 1 = 121 dots for computing
                            for(filter_t hKernel = 0; hKernel < KS_SIZE; hKernel++){
                            #pragma HLS UNROLL
                                for(filter_t wKernel = 0; wKernel < KS_SIZE-1; wKernel++){
                                #pragma HLS UNROLL
                                    //// left shift
                                    bufferPixKS[ciDots][hKernel][wKernel] = bufferPixKS[ciDots][hKernel][wKernel+1];
                                }
                                bufferPixKS[ciDots][hKernel][KS_SIZE-1] = pixChInColsKS[ciDiv][ciDots][hKernel];
                            }
                        }else{
                            winPixKS[ciDiv][ciDots].shift_pixels_left();
                            winPixKS[ciDiv][ciDots].insert_right_col(pixChInColsKS[ciDiv][ciDots]);
                        }
                    }
                    //bufferPixKS and winPixKS --> px5StrmValue;
                    if( (x >= 0) && (x<widthSize) ){
                        for(offset_t ciDots = 0; ciDots < PE_KS_CI; ciDots++ ){
                        #pragma HLS UNROLL
                            for(filter_t hKernel = 0; hKernel <  KS_SIZE; hKernel++){
                            #pragma HLS UNROLL
//                            	fprintf(outFile,"[%4d][%4d][%4d][%4d]",int(y),int(x),int(ciDiv),int(hKernel));
                                for(filter_t wKernel = 0; wKernel <  KS_SIZE; wKernel++){
                                #pragma HLS UNROLL
                                    offset_t index = ciDots * KS_SIZE * KS_SIZE + hKernel * KS_SIZE + wKernel;
                                    data_t bufPix;
                                    if( 1 == chInPEDiv ){
                                        //remK = (KS_SIZE-KERNEL)/2;////remainder
                                        //denK = KERNEL+remK;//denominator
                                    	//For example, KS_SIZE = 5, KERNEL =3
                                    	//remK = 1, denK=3+1=4;
                                    	//hKernel,wKernel
                                    	//00,01,02,03,04
                                    	//10,11,12,13,14
                                    	//20,21,22,23,24
                                    	//30,31,32,33,34
                                    	//40,41,42,43,44
                                    	//We hope 11,12,13,21,22,23,31,32,33 hold, others are zero.
                                    	if( ((hKernel%denK)<remK) || ((wKernel%denK)<remK)){
                                    		bufPix = 0;//virtual padSize,set 0
                                    	}else{
                                    		bufPix = bufferPixKS[ciDots][hKernel][wKernel];
                                    	}
                                    }else{
                                    	if( ((hKernel%denK)<remK) || ((wKernel%denK)<remK)){
                                    		bufPix = 0;//virtual padSize,set 0
                                    	}else{
                                    		bufPix = winPixKS[ciDiv][ciDots].val[hKernel][wKernel];
                                    	}
                                    }
//                                    fprintf(outFile,"%4d ",int(bufPix));
                                    hls::AXISetBitFields<PXS_STRM_WIDTH, data_t>(px5StrmValue, index * DOT_BITS , DOT_BITS, bufPix);
                                }
//                                fprintf(outFile,"\n");
                            }
                        }
                        //px5StrmValue --> hlsPxsStrm;
                        if(!skipPixel){
                            hlsPxsStrm << px5StrmValue;
                        }
                    }//End if x >=0
                }//End if y >= (KS_SIZE-PAD_SIZE)
            }//End ciDiv4
        }//End X
    }//End Y
#ifndef __SYNTHESIS__
    printf("Image max data: %4d, min data:%4d \n",short(maxData),short(minData));
#endif
}//End loadImageFromDRAM
//================================================================
//= getKSWeights
//================================================================
void getKSWeights(
        const channels_t coIndex                                                 ,
        const channels_t ciIndex                                                 ,
        apfix32_weights_t WBRAM[BLOCK_SIZE][PE_KS_CI][PE_KS_CO_PORT][SQR_KS_SIZE],
        data_t weightsS[PE_KS_CO][PE_KS_CI][SQR_KS_SIZE]
        ){
    #pragma HLS FUNCTION_INSTANTIATE variable = coIndex
    #pragma HLS FUNCTION_INSTANTIATE variable = ciIndex
    #pragma HLS inline
    #pragma HLS pipeline
    //calculate address
    //ci0co0,ci0co1,ci0co2,ci0co3,ci1co0,ci1co2,ci1co3.......
    offset_t coAddrOffsetCiDiv = (ciIndex/PE_KS_CI) * CEIL_DIV(CH_OUT,PE_KS_CO)*PE_KS_CO;
    block_size_t blkAddr = ( (coIndex + coAddrOffsetCiDiv)/(BRAM_1ADOTS * PE_KS_CO_PORT) ) % BLOCK_SIZE;
//    data_t weightsTmpL[PE_KS_CO][PE_KS_CI][SQR_KS_SIZE];
//	#pragma HLS ARRAY_PARTITION variable = weightsTmpL complete dim = 0
    ////For example
    ////Such KS_SIZE = 5, KERNEL = 3
    //// 0  1  2  3  4
    //// 5  6  7  8  9
    ////10  11 12 13 14
    ////15  16 17 18 19
    ////20  21 22 23 24
    //// so firstPos = 6,
    filter_t fristPos = (KS_SIZE - KERNEL)*(KS_SIZE+1)/2;
//    filter_t posIndex = fristPos;
    L_GETWEIGHTS_CIDIV:
    for(channels_t ci = 0; ci < PE_KS_CI; ci++){
        L_GETWEIGHTS_KERNEL:
        for(filter_t k = 0; k < SQR_KS_SIZE; k++){
				L_GETWEIGHTS_CODIV4:
				for(channels_t coDiv4 = 0; coDiv4 < PE_KS_CO_PORT; coDiv4++){
					data_t weightsValue[BRAM_1ADOTS];
					ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>(WBRAM[blkAddr][ci][coDiv4][k],weightsValue);
#ifndef __SYNTHESIS__
//					if(1 == coDiv4){
//						fprintf(outFile,"[ %4d %4d %4d ]=%4d \n",int(ci),int(k),int(coDiv4),int(weightsValue[2]));
//					}
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
#endif
					// we only write the center valid weights.
					weightsS[coDiv4 * BRAM_1ADOTS + 0][ci][k] = weightsValue[0];
					weightsS[coDiv4 * BRAM_1ADOTS + 1][ci][k] = weightsValue[1];
					weightsS[coDiv4 * BRAM_1ADOTS + 2][ci][k] = weightsValue[2];
					weightsS[coDiv4 * BRAM_1ADOTS + 3][ci][k] = weightsValue[3];
				}
        }
    }
}

void macc2dKS(
        const data_t pixels5[SQR_KS_SIZE]   ,
        const data_t weights5[SQR_KS_SIZE]  ,
            result_t &result ) {
#pragma HLS inline
    result_t accumulator = 0;
    data_result_t mulResult[SQR_KS_SIZE];
    #pragma HLS ARRAY_PARTITION variable = mulResult complete dim = 0

    #ifndef __SYNTHESIS__
//                m = i * KL_SIZE;
//    	    fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d ] \n",
//    			  (int)pixels5[m+0], (int)pixels5[m+1], (int)pixels5[m+2], (int)pixels5[m+3], (int)pixels5[m+4],
//    			  (int)pixels5[m+5], (int)pixels5[m+6], (int)pixels5[m+7], (int)pixels5[m+8], (int)pixels5[m+9],
//    			  (int)pixels5[m+10] );
//          }
//          for(int i=0;i<KL_SIZE;i++){
//                m = i * KL_SIZE;
//             fprintf(outFile,"Weights:[ %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d ] \n",
//  			  (int)weights5[m+0], (int)weights5[m+1], (int)weights5[m+2], (int)weights5[m+3], (int)weights5[m+4],
//  			  (int)weights5[m+5], (int)weights5[m+6], (int)weights5[m+7], (int)weights5[m+8], (int)weights5[m+9],
//  			  (int)weights5[m+10] );
//          }
//      }
   #endif
#if 1
    L_MACC_MULTIPLY:
    for(filter_t i = 0; i< SQR_KS_SIZE; i++){
    #pragma HLS UNROLL
        mulResult[i] = pixels5[i] * weights5[i] ;
    }
    L_MACC_ACCUMULATE:
    for(filter_t i = 0; i< SQR_KS_SIZE; i++){
    #pragma HLS UNROLL
       accumulator = accumulator + result_t(mulResult[i]);
    }
    result = accumulator ;
#else
    result = 0;
#endif
}

//================================================================
//= computeElementKS
//================================================================
void computeElementKS(
        hls_pxs_strm_t &hlsPxsStrm                                                   ,
        hls_outs_strm_t &hlsOutlStrm                                                 ,
        apfix32_weights_t WBRAM_KS[BLOCK_SIZE][PE_KS_CI][PE_KS_CO_PORT][SQR_KS_SIZE]
        ){
#pragma HLS inline off
    // write here by cqiu
    result_t OUTBRAM[MAX_NUM_SPLIT_CHOUT/PE_KS_CO][PE_KS_CO];
    #pragma HLS ARRAY_PARTITION variable = OUTBRAM complete dim = 2
    #pragma DEPENDENCE variable = OUTBRAM inter flase
    result_t OUTBRAM_BUF0[PE_KS_CO];
    #pragma HLS ARRAY_PARTITION variable = OUTBRAM_BUF0 complete dim = 0
    pxs_strm_t pixelValue;
    data_t pixelsS[PE_KS_CI][SQR_KS_SIZE];
    #pragma HLS ARRAY_PARTITION variable = pixelsS complete dim = 0
    offset_t ciOffset;
    channels_t chInPEDiv = CEIL_DIV(CH_IN,PE_KS_CI);
    channels_t chOutPEDiv = CEIL_DIV(CH_OUT,PE_KS_CO);
    // 11x11 deal with 7x7,PAD_SIZE=0, so padSizeExtend = 2
    filter_t remK = (KS_SIZE-KERNEL)/2;////remainder
    dimension_t padSizeExtend = PAD_SIZE + dimension_t(remK);
    dimension_t widthSize = STRIDE*(1+(WIDTH_IN+2*padSizeExtend-KS_SIZE)/STRIDE);
    dimension_t heightSize = STRIDE*(1+(HEIGHT_IN+2*padSizeExtend-KS_SIZE)/STRIDE);
    L_COMPUTE_Y:
    for(dimension_t y = 0; y < KS_HEIGHT_STRIDE; y++){
    #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16
        L_COMPUTE_X:
        for(dimension_t x = 0; x < widthSize; x++){
        #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16
            L_COMPUTE_CIDIV4:
            for(channels_t ciDiv = 0; ciDiv < chInPEDiv ; ciDiv++){
            #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16
                L_PARALLEL_CODIV8:
                for(channels_t coDiv = 0; coDiv < chOutPEDiv; coDiv++){
                #pragma HLS LOOP_TRIPCOUNT min = 1 max = 32 avg = 16
                #pragma HLS pipeline II = 1
                    bool skipPixel = (STRIDE == 2) & (x%2 | y%2);
                    if(!skipPixel){
                        //Load pixels5
                        if(coDiv == 0){
                            hlsPxsStrm >> pixelValue;
                            //Load pixels5
                            L_LOAD_PIXELS_CI_PE:
                            for(channels_t ciDot = 0; ciDot < PE_KS_CI; ciDot++){
                            #pragma HLS unroll
                                L_LOAD_PIXEL_KS_PE:
                                for(channels_t k = 0; k < SQR_KS_SIZE; k++){
                                #pragma HLS unroll
                                    offset_t index = ciDot * SQR_KS_SIZE + k;
                                    hls::AXIGetBitFields<PXL_STRM_WIDTH,data_t>(pixelValue,index * DOT_BITS, DOT_BITS, pixelsS[ciDot][k]);
                                }//End k
								#ifndef __SYNTHESIS__
//								  if(CH_IN ==3){
////									  fprintf(outFile,"x,y,ciDiv=%4d,%4d,%4d\n",int(x),int(y),int(ciDiv));
//									  int m = 0;
//									  for(int i=0;i<KS_SIZE;i++){
//											m = i * KS_SIZE;
//											if(x/2 > 122){
//												fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d ] \n",
//													  (int)pixelsL[ciDot][m+0], (int)pixelsL[ciDot][m+1], (int)pixelsL[ciDot][m+2], (int)pixelsL[ciDot][m+3], (int)pixelsL[ciDot][m+4],
//													  (int)pixelsL[ciDot][m+5], (int)pixelsL[ciDot][m+6], (int)pixelsL[ciDot][m+7], (int)pixelsL[ciDot][m+8], (int)pixelsL[ciDot][m+9],
//													  (int)pixelsL[ciDot][m+10] );
//											}
//									  }
//								  }
								#endif
                            }//End ciDot
//                            ciOffset = ciDiv4 * CEIL_DIV(CH_OUT,BRAM_1ADOTS) * BRAM_1ADOTS;
                        }//End coDiv8 ==0
                        //Load weightsL
                        data_t weightsS[PE_KS_CO][PE_KS_CI][SQR_KS_SIZE];
                        #pragma HLS ARRAY_PARTITION variable = weightsS complete dim = 0
                        getKSWeights( coDiv * PE_KS_CO ,ciDiv * PE_KS_CI, WBRAM_KS,weightsS);

                        //Compute parallel
                        L_COMPUTE_CO_MUL:
                        for(channels_t coMul = 0; coMul < PE_KS_CO/2; coMul++ ){
                        #pragma HLS UNROLL
                            result_t result[2][PE_KS_CI];
							#pragma HLS ARRAY_PARTITION variable = result complete dim = 0
                            L_COMPUTE_CI_MUL:
                            for(channels_t ciMul = 0; ciMul < PE_KS_CI; ciMul++){
                            #pragma HLS UNROLL
                            	macc2dKS(pixelsS[ciMul],weightsS[2*coMul][ciMul],result[0][ciMul]);
                            	macc2dKS(pixelsS[ciMul],weightsS[channels_t(2*coMul+1)][ciMul],result[1][ciMul]);
//                            	macc2dKS_int8(pixelsL[ciMul],weightsL[2*coMul][ciMul],weightsL[channels_t(2*coMul+1)][ciMul],result[0][ciMul],result[1][ciMul] );
                            }
                            result_t resultSum[2];
							#pragma HLS ARRAY_PARTITION variable = resultSum complete dim = 0
                            L_COMPUTE_CO_MUL_DOUBLE:
                            for(channels_t coDouble = 0; coDouble < 2; coDouble++ ){
                            #pragma HLS UNROLL
//
							    resultSum[coDouble] = result[coDouble][0] + result[coDouble][1] + result[coDouble][2] + result[coDouble][3];
//
//                            	resultSum[coDouble] = result[coDouble][ciDiv%PE_KS_CI];

								if(CEIL_DIV(CH_OUT,PE_KS_CO) == 1){
									if(ciDiv == 0){
										OUTBRAM_BUF0[2*coMul+coDouble] = resultSum[coDouble];
									}else{
										OUTBRAM_BUF0[2*coMul+coDouble] = OUTBRAM_BUF0[2*coMul+coDouble] + resultSum[coDouble];
									}
								}else{
									if(ciDiv == 0){
										OUTBRAM[coDiv][2*coMul+coDouble] = resultSum[coDouble];
									}else{
										OUTBRAM[coDiv][2*coMul+coDouble] = OUTBRAM[coDiv][2*coMul+coDouble] + resultSum[coDouble];
									}
								}
#ifndef __SYNTHESIS__
//									if(6 ==int( 2*coMul+coDouble + coDiv8 * 8)){
//										fprintf(outFile,"y%4d,x%4d,ci=%4d,d=%8f\n",int(y),int(x),int(ciDiv),
//												int(OUTBRAM[coDiv8][2*coMul+coDouble]));
//									}
#endif
                            }
                        }
                        //Sent Result
                        if(ciDiv == chInPEDiv-1 ){
                            outs_strm_t outLStrmValue;
                            result_t outData[PE_KS_CO];
                            #pragma HLS ARRAY_PARTITION variable = outData complete dim = 0
                            //Transmit the result to outData;
                            L_COMPUTE_POST:
                            for(channels_t coPost = 0 ; coPost < PE_KS_CO; coPost++){
                            #pragma HLS UNROLL
                                if(CEIL_DIV(CH_OUT,PE_KS_CO) == 1){
                                    outData[coPost] = OUTBRAM_BUF0[coPost];
                                }else{
                                    outData[coPost] = OUTBRAM[coDiv][coPost];
                                }
                            }
                            L_COMPUTE_TRANSMIT_TO_STRM:
                            for(emem_1adots_t ememCnt = 0; ememCnt < PE_KS_CO; ememCnt++){
	                        #pragma HLS unroll

                            	if(outData[ememCnt] >= (1 << (2 * DOT_BITS-1)) ){
                            		outData[ememCnt] = (1 << (2 * DOT_BITS -1)) - 1 ;
                            	}

                            	if(outData[ememCnt] < (-(1 << (2 * DOT_BITS-1))) ){
                            		outData[ememCnt] = -(1 << (2 * DOT_BITS -1));
                            	}
                                hls::AXISetBitFields(outLStrmValue,int(ememCnt * DOT_BITS * 2), int(DOT_BITS *2) , data_result_t(outData[ememCnt]));
								#ifndef __SYNTHESIS__
//									fprintf(outFile,"y=%4d,x=%4d,coDiv8=%4d,outdata=%8f\n",
//											int(y),int(x),int(coDiv8*PE_KS_CO+ememCnt),float( (outData[ememCnt])/(128.0f)));
//									fprintf(outFile,"%8f\n",float((outData[ememCnt])/(128.0f)));
								#endif
                            }
                            hlsOutlStrm << outLStrmValue;
                        }// End Compute
                    }//End skipPixel
                }//End coDiv8
            }//End ciDiv4
        }//End x
    }//End y
}
//================================================================
//= writeBackToDRAMKS
//================================================================
void writeBackToDRAMKS(
        hls_outs_strm_t &hlsOutsStrm,
        iemem_t *WRITE_SHARED_DRAM,
		apfix32_weights_t BBRAM_KS[BLOCK_SIZE][BIAS_NUM][PE_KS_CO_PORT_DIV4][WEMEM_1ADOTS_DIV4]
    ){
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable = LBRAM_KS complete dim = 2 // EXTMEM_ALIGNMENT
#pragma HLS RESOURCE variable = LBRAM_KS core = RAM_T2P_BRAM latency = 3
//data_t LBRAM[MAX_WIDTH_X_CHIN_DIV16][WEMEM_1ADOTS];////512x16
data_t maxData = 0;
data_t minData = 255;
//dimension_t widthOut = (STRIDE ==2) ? dimension_t(WIDTH_IN/2) : WIDTH_IN ;
num_sublayers_t subLayerMulti = (SUB_LAYER_FLAG == 0) ? num_sublayers_t(1) : SUB_LAYER_NUM   ;
offset_t xOffsetDiv16 = subLayerMulti * CEIL_DIV(CH_OUT,IEMEM_1ADOTS);
offset_t yOffsetDiv16 = xOffsetDiv16 * KS_WIDTH ;
result_t halfValueCaffe = 1<<(IMG_POS_IN + WEIGHTS_POS - IMG_POS_OUT-1);
result_t halfValueYolo  = 1<<(IMG_POS_IN + WEIGHTS_POS + SCALES_POS - IMG_POS_OUT-1);

    L_WRITEBACK_Y:
    for(dimension_t y = 0 ; y < KS_HEIGHT + 1; y++){
    #pragma HLS LOOP_TRIPCOUNT MIN = 8 AVG = 16 MAX = 32
        L_WRITEBACK_X:
        for(dimension_t x = 0; x < KS_WIDTH; x++){
        #pragma HLS LOOP_TRIPCOUNT MIN = 8 AVG = 16 MAX = 32
            L_WRITEBACK_CODIV16:
            for(channels_t coDiv16 = 0; coDiv16 < CEIL_DIV(CH_OUT,IEMEM_1ADOTS); coDiv16++){
            #pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 2 MAX = 4
            #pragma HLS pipeline II = 1
                data_result_t raw[IEMEM_1ADOTS];
                result_t bias[IEMEM_1ADOTS];
                result_t mean[IEMEM_1ADOTS];
                result_t variance[IEMEM_1ADOTS];
                result_t batchData[IEMEM_1ADOTS];
                result_t batchDataRelu[IEMEM_1ADOTS];
                result_t batchDataDiv10Int[IEMEM_1ADOTS];
                data_t activeData[IEMEM_1ADOTS];
                #pragma HLS ARRAY_PARTITION variable = raw complete dim = 0
                #pragma HLS ARRAY_PARTITION variable = variance complete dim = 0
                #pragma HLS ARRAY_PARTITION variable = mean complete dim = 0
                #pragma HLS ARRAY_PARTITION variable = bias complete dim = 0
                #pragma HLS ARRAY_PARTITION variable = batchData complete dim = 0
                #pragma HLS ARRAY_PARTITION variable = batchDataRelu complete dim = 0
                #pragma HLS ARRAY_PARTITION variable = batchDataDiv10Int complete dim = 0
                #pragma HLS ARRAY_PARTITION variable = activeData complete dim = 0
                data_t preX[MAX_CHANNELS/IEMEM_1ADOTS][IEMEM_1ADOTS];
                data_t curX[MAX_CHANNELS/IEMEM_1ADOTS][IEMEM_1ADOTS];
                data_t maxX[MAX_CHANNELS/IEMEM_1ADOTS][IEMEM_1ADOTS];
                data_t tmpX[MAX_CHANNELS/IEMEM_1ADOTS][IEMEM_1ADOTS];
                #pragma HLS ARRAY_PARTITION variable = preX complete dim = 2
                #pragma HLS ARRAY_PARTITION variable = curX complete dim = 2
                #pragma HLS ARRAY_PARTITION variable = maxX complete dim = 2
                #pragma HLS ARRAY_PARTITION variable = tmpX complete dim = 2
                memaddr_t ddrAddr ;
                iemem_t iememTmp;
                //The last row process
                if( y == KS_HEIGHT ){
                    if(POOLING_TYPE == 1){// Max Pooling Stride == 1
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + (y-1) * yOffsetDiv16 + x * xOffsetDiv16 + coDiv16;
                        ApFixToExtMemSync<DOT_BITS,iemem_t,data_t,IEMEM_1ADOTS>(iememTmp,LBRAM_KS[offset_t(x * xOffsetDiv16  + coDiv16)]);
                    }
                }else{
                //Normal rows process
                    //Load raws values
                    outs_strm_t outResult;
                    if( 0 == (coDiv16%4)){
                        hlsOutsStrm >> outResult;
                        ExtMemToApFixSync<DOT_BITS * 2 ,outs_strm_t,data_result_t,PE_KS_CO>(outResult,raw);
                    }


                    data_t apfixTmp[3][PE_KS_CO_PORT][BRAM_1ADOTS];
                    //Load bias to apfixTmp[0]
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                    	(BBRAM_KS[coDiv8/2][0][(coDiv16/4)%4][coDiv16%4],apfixTmp[0][0]);

//                    BBRAM_KS[BLOCK_SIZE][BIAS_NUM][PE_KS_CO_PORT_DIV4][WEMEM_1ADOTS_DIV4]
//
//                        (BBRAM_KS[coDiv8/2][0][coDiv8%2][0],apfixTmp[0][0]);
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_KS[coDiv8/2][0][coDiv8%2][1],apfixTmp[0][1]);
//
//                    //Load means to apfixTmp[1]
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_KS[coDiv8/2][2][coDiv8%2][0],apfixTmp[1][0]);
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_KS[coDiv8/2][2][coDiv8%2][1],apfixTmp[1][1]);
//                    //Load variance to apfixTmp[2]
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_KS[coDiv8/2][3][coDiv8%2][0],apfixTmp[2][0]);
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_KS[coDiv8/2][3][coDiv8%2][1],apfixTmp[2][1]);
                    for(channels_t coIndex = 0; coIndex < IEMEM_1ADOTS; coIndex++){
                    #pragma HLS UNROLL factor = IEMEM_1ADOTS
                    	if( 0 == (coIndex%4)) {
                            //Load bias to apfixTmp[0]
                             ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
                             	 (BBRAM_KS[coDiv16/PE_KS_CO_PORT_DIV4][0][coDiv16%PE_KS_CO_PORT_DIV4][coIndex/PE_KS_CO_PORT_DIV4],apfixTmp[0][coIndex/PE_KS_CO_PORT_DIV4]);

                    	}
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
                            activeData[coIndex] = data_result_t( (batchDataRelu[coIndex]+halfValueYolo) >> (IMG_POS_IN + WEIGHTS_POS + SCALES_POS - IMG_POS_OUT));
                        }else{

                            batchData[coIndex] = raw[coIndex] + (bias[coIndex] << (IMG_POS_IN + WEIGHTS_POS - BIAS_POS));
                            batchDataRelu[coIndex] = (batchData[coIndex] > 0) ? batchData[coIndex] : result_t(0) ;
//                            batchDataRelu[coIndex] =  batchData[coIndex] ;
                            activeData[coIndex] = data_result_t( (batchDataRelu[coIndex]+halfValueCaffe ) >> (IMG_POS_IN + WEIGHTS_POS - IMG_POS_OUT));
							#ifndef __SYNTHESIS__
//                            		int valTimes=1;
//                            	if(IMG_POS_OUT < 0){
//                            		valTimes = 1 << (-IMG_POS_OUT);
//                            		fprintf(outFile,"%8f\n",float(activeData[coIndex]*1.0f * valTimes ));
//                            	}else{
//                            		valTimes = 1 << (IMG_POS_OUT);
//                            		fprintf(outFile,"%8f\n",float(activeData[coIndex]*1.0f / valTimes ));
//                            	}
							#endif
                        }
                        if(x == 0){
                            preX[coDiv16][coIndex] = -MAX_SHORT;
                        } else {
                            preX[coDiv16][coIndex] = curX[coDiv16][coIndex];
                        }
                        if(POOLING_TYPE == 2){// Max Pooling Stride == 2
                            if( (!(y%2)) && (x%2) ) {// even row, odd column
                                maxX[coDiv16][coIndex] = max(preX[coDiv16][coIndex],activeData[coIndex]);
                                LBRAM[ x * xOffsetDiv16  + coDiv16][coIndex] = maxX[coDiv16][coIndex];
                            } else if ( (y%2) && (x%2)) {// odd row, odd column
                                tmpX[coDiv16][coIndex] = max(LBRAM[x * xOffsetDiv16  + coDiv16][coIndex],preX[coDiv16][coIndex]);
                                maxX[coDiv16][coIndex] = max(tmpX[coDiv16][coIndex],activeData[coIndex]);
                            }
                        } else if(POOLING_TYPE == 1){ // Max Pooling Stride == 1
                            if(y == 0){
                                maxX[coDiv16][coIndex] = max(preX[coDiv16][coIndex],activeData[coIndex]);
                                LBRAM[ x * xOffsetDiv16  + coDiv16][coIndex] = maxX[coDiv16][coIndex];
                            }else{
                                tmpX[coDiv16][coIndex] = max(activeData[coIndex],preX[coDiv16][coIndex]);
                                maxX[coDiv16][coIndex] = max(tmpX[coDiv16][coIndex],data_t(LBRAM[ x * xOffsetDiv16  + coDiv16][coIndex]));
                                LBRAM[ x * xOffsetDiv16  + coDiv16][coIndex] = tmpX[coDiv16][coIndex];
                            }
                        }
                        curX[coDiv16][coIndex] = activeData[coIndex];
                    }//End coIndex


                    if(POOLING_TYPE == 2){// Max Pooling Stride == 2
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + (y/2) * yOffsetDiv16 / 2 + (x/2) * xOffsetDiv16 + coDiv16;
                        ApFixToExtMemSync<DOT_BITS,iemem_t,data_t,IEMEM_1ADOTS>(iememTmp,maxX[coDiv16]);
                    }else if(POOLING_TYPE == 0){// No Max Pooling
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + y * yOffsetDiv16  + x * xOffsetDiv16 + coDiv16;
                        ApFixToExtMemSync<DOT_BITS,iemem_t,data_t,IEMEM_1ADOTS>(iememTmp,activeData);

                    }else if(POOLING_TYPE == 1){// Max Pooling Stride == 1
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + (y-1) * yOffsetDiv16  + x * xOffsetDiv16 + coDiv16;
                        ApFixToExtMemSync<DOT_BITS,iemem_t,data_t,IEMEM_1ADOTS>(iememTmp,maxX[iememTmp]);
                    }
                }//End Normal rows
                if( (   (y != KS_HEIGHT) &&
                        (((POOLING_TYPE == 2) && (y%2) && (x%2)) || (POOLING_TYPE == 0)) )
                   || ((POOLING_TYPE == 1) && (y != 0)) ){


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
                    		int valTimes=1;
                    		//// conv1 check ok
							if(IMG_POS_OUT < 0){
								valTimes = 1 << (-IMG_POS_OUT);
								fprintf(outFile,"%4d\n",int(resultValue[rCnt] * valTimes ));
							}else{
								valTimes = 1 << (IMG_POS_OUT);
								fprintf(outFile,"%4d\n",int(resultValue[rCnt] / valTimes ));
							}
						}
					#endif

                }
            }//End coDiv8
        }//End x
    }//End y
#ifndef __SYNTHESIS__
    printf("PostProcess maxOutData %4d,minOutData %4d \n", int(maxData), int(minData));
#endif
}//End writeBackToDRAM
//================================================================
//= dataFlowProcess
//================================================================
void dataFlowProcessKS(
        wemem_t *WEIGHTS_SHARED_DRAM,
        iemem_t *READ_SHARED_DRAM,
        iemem_t *WRITE_SHARED_DRAM
		){
#pragma HLS inline off
#pragma HLS DATAFLOW
    hls_pxs_strm_t hlsPxsStrm    ;
    hls_outs_strm_t hlsOutsStrm	 ;
#pragma HLS STREAM variable = hlsPxsStrm depth = 512 dim = 1
#pragma HLS STREAM variable = hlsOutsStrm depth = 512 dim = 1
    loadImageFromDRAMKS(
       READ_SHARED_DRAM	,
       hlsPxsStrm
       );
   computeElementKS(
       hlsPxsStrm       ,
       hlsOutsStrm      ,
       WBRAM_KSA
       );
   writeBackToDRAMKS(
	   hlsOutsStrm      ,
       WRITE_SHARED_DRAM,
       BBRAM_KSA
       );
}


















//================================================================================================================================
//================================================================================================================================
//================================================================================================================================
//================================================================================================================================
//================================================================================================================================
//================================================================================================================================
//================================================================
//= dataFlowProcess
//================================================================
void dataFlowProcessKL(
        wemem_t *WEIGHTS_SHARED_DRAM,
        iemem_t *READ_SHARED_DRAM,
        iemem_t *WRITE_SHARED_DRAM
		){
#pragma HLS inline off
#pragma HLS DATAFLOW
    hls_pxl_strm_t hlsPxlStrm    ;
    hls_outl_strm_t hlsOutlStrm	;
#pragma HLS STREAM variable = hlsPxlStrm depth = 512 dim = 1
#pragma HLS STREAM variable = hlsOutlStrm depth = 512 dim = 1
    loadImageFromDRAM(
       READ_SHARED_DRAM	,
       hlsPxlStrm
       );
   computeElement(
       hlsPxlStrm       ,
       hlsOutlStrm      ,
       WBRAM_KLA
       );
   writeBackToDRAM(
	   hlsOutlStrm      ,
       WRITE_SHARED_DRAM,
       BBRAM_KLA
       );
}
//================================================================
//= loadWeightsFromDRAMKL
//================================================================
void loadWeightsFromDRAMKL(
        wemem_t *SHARED_DRAM,
        apfix32_weights_t WBRAM_KL[BLOCK_SIZE][PE_KL_CI][PE_KL_CO_PORT][SQR_KL_SIZE],
        apfix32_weights_t BBRAM_KL[BLOCK_SIZE][BIAS_NUM][WEMEM_1ADOTS_DIV8][PE_KL_CO_PORT]
	){
#pragma HLS inline off

#pragma HLS ARRAY_PARTITION variable = WBRAM_KLA complete dim = 2 //ci
#pragma HLS ARRAY_PARTITION variable = WBRAM_KLA complete dim = 3 //co
#pragma HLS ARRAY_PARTITION variable = WBRAM_KLA complete dim = 4   //k
#pragma HLS RESOURCE variable = WBRAM_KLA core = RAM_S2P_BRAM latency = 3

#pragma HLS ARRAY_PARTITION variable = BBRAM_KLA complete dim = 2 //biasNum
#pragma HLS ARRAY_PARTITION variable = BBRAM_KLA complete dim = 3 //coDiv8
#pragma HLS ARRAY_PARTITION variable = BBRAM_KLA complete dim = 4 //PE_KL_CO_PORT
#pragma HLS RESOURCE variable = BBRAM_KLA core = RAM_S2P_BRAM latency = 3
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
            ExtMemToApFixSync<BRAM_WIDTH,wemem_t,apfix32_weights_t,WEMEM_WIDTH/BRAM_WIDTH>(biasData,apfix32Bias);
            for(emem_1adots_t m = 0; m < WEMEM_1ADOTS_DIV4; m++){
	        #pragma HLS unroll
            	BBRAM_KL[coDiv16][biasId][m/PE_KL_CO_PORT][m%PE_KL_CO_PORT] = apfix32Bias[m];
//            	#ifndef __SYNTHESIS__
//            	    fprintf(outFile,"bias %16x \n",(int)apfix32Bias[m]);
//            	#endif
            }
        }
    }
    //Load weights coefficients
    addrBaseOffset = (WEIGHTS_OFFSET+LAYER_WEIGHTS_OFFSET) ;
    filter_t fristPos = (KL_SIZE - KERNEL)*(KL_SIZE+1)/2;
    L_LOAD_WEIGHTS_FROM_DRAM:
    for (channels_t ci =0; ci < nextChIn; ci++ ){
    #pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 2 MAX = 16
    	dramAddr = 0;
        memaddr_t addrOffset = addrBaseOffset + ci * filterMulChOut * DOT_BYTES / SIZEOF_WEMEM ;
        memcpy(WBuff,&SHARED_DRAM[addrOffset],filterMulChOut * DOT_BYTES);
        L_LOAD_FILTERS_FROM_DRAM:
        for (filter_t k = 0; k < filterSize; k++){
        #pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 49 MAX = 121
            L_LOAD_CHOUTDIV_FROM_DRAM:
            for(channels_t coDiv8 = 0; coDiv8 < nextChOutDiv16*2; coDiv8++){
                #pragma HLS PIPELINE II = 1
                //First 	: Traverse co , from 0 to nextChOut
                //Second  	: Traverse k  , from 0 to 120 (nextKernel=11)
                //Third		: Traverse ci , from 0 to nextChIn
                apfix32_weights_t apfix32Weights[WEMEM_1ADOTS_DIV4];
                #pragma HLS ARRAY_PARTITION variable = apfix32Weights complete dim = 0
                if(!(coDiv8%2)){
                    wemem_t weights = WBuff[dramAddr++];
                    ExtMemToApFixSync<BRAM_WIDTH,wemem_t,apfix32_weights_t,WEMEM_WIDTH/BRAM_WIDTH>(weights,apfix32Weights);
                }
                //Calucate the Address
                channels_t ciId = ci % PE_KL_CI;
                //case 0  filterSize=11x11,sqrKTimes=1;
                //case 1  filterSize=9x9,sqrKTimes=1,reserve;
                //case 2  filterSize=7x7,sqrKTimes=2;
                //case 3  filterSize=5x5,sqrKTimes=4;
//                filter_t sqrKTimes = FLOOR_DIV(filterSize,MAX_FILTER_SIZE);
                block_size_t blkSizeId = (coDiv8 * 8 + (ci/PE_KL_CI) * nextChOut )/(BRAM_1ADOTS * PE_KL_CO_PORT );
                emem_1adots_t baseAddr = (coDiv8 % PE_KL_CO_PORT ) * PE_KL_CO_PORT;
                filter_t kId = fristPos+(k%KERNEL)+(k/KERNEL)*KL_SIZE;
//				weightsL[coDiv4 * BRAM_1ADOTS + 0][ci][fristPos+(k%KERNEL)+(k/KERNEL)*KL_SIZE] = weightsValue[0];

                //Write weights to WBRAM_KL
                for(emem_1adots_t m = 0; m < PE_KL_CO_PORT; m++){
                #pragma HLS unroll
                	WBRAM_KL[blkSizeId][ciId][m][kId] = apfix32Weights[baseAddr+m];
//					#ifndef __SYNTHESIS__
//						fprintf(outFile,"wbram %16x \n",(int)apfix32Weights[baseAddr+m]);
//					#endif
                }
//#ifndef __SYNTHESIS__
//	data_t tmpW[BRAM_1ADOTS];
//	ExtMemToApFixSync<DOT_BITS, apfix32_weights_t,data_t,BRAM_1ADOTS >(WBRAM_KL[blkSizeId][ciId][1][kId],tmpW);
//	fprintf(outFile,"ci,k [%4d][%4d]=%4d \n",int(ci),int(k),(int)tmpW[2]);
//#endif
                //                if(6 == co)
                //                	printf("ci,kernel,co=[%4d],[%4d],[%4d]::%4d \n",ci,kernel,co,int(dataTmp));
            }

        }
    }
}
//================================================================
//= loadImageFromDRAM
//================================================================
void loadImageFromDRAM(
    iemem_t *READ_SHARED_DRAM	,
    hls_pxl_strm_t &hlsPxlStrm
    ){
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable = IBRAM_KL complete dim = 2 //y
#pragma HLS ARRAY_PARTITION variable = IBRAM_KL complete dim = 3 //chin
#pragma HLS RESOURCE variable = IBRAM_KL core = RAM_S2P_BRAM latency = 3
    printf(" LoadImageFromDram Function data path                                 \n");        
    printf(" READ_SHARED_DRAM --> IBuff --> IBRAM_KL --> pixChInColsKL            \n");
    printf(" --> winPixKL and bufferPixKL --> px5StrmValue -->hlsPxlStrm          \n");
#ifndef __SYNTHESIS__
    data_t maxData = 0;
    data_t minData = 127;
#endif
    //IBuff
	iemem_t IBuff[MAX_WIDTH_X_CHIN_DIV16_KL];
    offset_t lineWidthDiv16 = CEIL_DIV(DDR_CH_IN,IEMEM_1ADOTS) * WIDTH_IN ;
	data_t pixChInColsKL[MAX_NUM_SPLIT_CHOUT/PE_KL_CI][PE_KL_CI][KL_SIZE];
	#pragma HLS ARRAY_PARTITION variable = pixChInColsKL complete dim = 2
	#pragma HLS ARRAY_PARTITION variable = pixChInColsKL complete dim = 3
	win_pix_kl_t winPixKL[MAX_NUM_SPLIT_CHOUT/PE_KL_CI][PE_KL_CI];
	#pragma HLS ARRAY_PARTITION variable = winPixKL complete dim = 2
	data_t bufferPixKL[PE_KL_CI][KL_SIZE][KL_SIZE];
	#pragma HLS ARRAY_PARTITION variable = bufferPixKL complete dim = 0
    channels_t chInPEDiv = CEIL_DIV(CH_IN,PE_KL_CI);
    //we use cover child algorithm.
    //Every 5x5,7x7,11x11 cnn, we all use 11x11
    //For example KL_SIZE = 5, KERNEL = 3,so pad is (5-3)/2 = 1
    // . . . . .
    // . x x x .
    // . x x x .
    // . x x x .
    // . . . . .
    // 11x11 deal with 7x7,PAD_SIZE=0, so padSizeExtend = 2
    filter_t remK = (KL_SIZE-KERNEL)/2;////remainder
    dimension_t padSizeExtend = PAD_SIZE + dimension_t(remK);
    filter_t denK = KERNEL+remK;//denominator
    dimension_t widthSize = STRIDE*(1+(WIDTH_IN+2*padSizeExtend-KL_SIZE)/STRIDE);
    dimension_t heightSize = STRIDE*(1+(HEIGHT_IN+2*padSizeExtend-KL_SIZE)/STRIDE);
    L_LOAD_IMAGES_FROM_DRAM_Y:
	// For example KL_SIZE-padSizeExtend = 9, y=0,1,2,3...8 , we should store  the y windows.
    for(dimension_t y = 0; y < heightSize + KL_SIZE-padSizeExtend ; y++  ){
    #pragma HLS LOOP_TRIPCOUNT min = 16 max = 269 avg = 20
        //Load One line width * chin images to IBuff
        if( y < HEIGHT_IN ){
            memaddr_t dramAddr = IMAGE_OFFSET + LAYER_IN_IMAGE_OFFSET + y * lineWidthDiv16;
            memcpy(IBuff,&READ_SHARED_DRAM[dramAddr],sizeof(iemem_t) * lineWidthDiv16);
        }
        //reset the bufAddr;
        offset_t bufAddr = 0;
        L_LOAD_IMAGES_FROM_DRAM_X:
		// For example KL_SIZE = 11, x=-10,-9,-8,-7... we should store the x windows.
        for(dimension_t x = -(KL_SIZE-1); x < WIDTH_IN; x++){
        #pragma HLS LOOP_TRIPCOUNT min = 12 max = 36 avg = 20
            L_LOAD_IMAGES_FROM_DRAM_CIDIV4:
            for(dimension_t ciDiv = 0; ciDiv < chInPEDiv ; ciDiv++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 512 avg = 8
            #pragma HLS pipeline II = 1
                //Write to IBRAM_KL
                if( (y < HEIGHT_IN) && (x >= 0) && ( ( (ciDiv*PE_KL_CI) % IEMEM_1ADOTS_DIV4 )==0) ){
                    apfix32_image_t vIBRAM[BRAM_1ADOTS];
	                #pragma HLS ARRAY_PARTITION variable = vIBRAM complete dim = 0
                    if( 0 == ((ciDiv * PE_KL_CI  ) % IEMEM_1ADOTS) ){
                        iemem_t vDatas = IBuff[bufAddr++];
                        ExtMemToApFixSync<BRAM_WIDTH, iemem_t,apfix32_image_t,IEMEM_1ADOTS_DIV4 >(vDatas,vIBRAM);
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
//									(short)vData[15],(short)vData[14],(short)vData[13],(short)vData[12],
//									(short)vData[11],(short)vData[10],(short)vData[9],(short)vData[8],
//									(short)vData[7],(short)vData[6],(short)vData[5],(short)vData[4],
//									(short)vData[3],(short)vData[2],(short)vData[1],(short)vData[0]
//								);
						#endif
                    }
                    offset_t yLineId = y % NUM_IMG_CACHE_LINES_KL;
                    //// here is wrong
                    offset_t xLineId = ( (x * DDR_CH_IN + ciDiv * PE_KL_CI)/(BRAM_1ADOTS * PE_KL_CI) ) % MAX_WIDTH_X_CHIN_DIV4_KL;
                    L_WRITE_TO_IBRAM_KL:
                    for(offset_t i = 0; i < PE_KL_CI ; i++ ){
                    #pragma HLS unroll
                        IBRAM_KL[xLineId][yLineId][i] = vIBRAM[i+((ciDiv*PE_KL_CI)%WEMEM_1ADOTS_DIV4)];
                    }
                }
                bool skipPixel = (STRIDE == 2) & ((x % 2) | ( (y-KL_SIZE+padSizeExtend) %2));
                // Valid methord, Write to pixChInColsKL
                if( y >= (KL_SIZE-padSizeExtend) ){
                    pxl_strm_t px5StrmValue;
                    // KL_SIZE-1-padSizeExtend=8
                    //        x=-10,-9,-8,-7,-6,....-2,-1,0,1, 2,....WIDTH_IN-1
                    //pixValidX= -2,-1, 0, 1, 2, 3,..6, 7,8,9,10,....WIDTH_IN+7
                    dimension_t pixValidX = x + (KL_SIZE-1-padSizeExtend);
                    offset_t iAddr = (pixValidX < 0)?offset_t(0):offset_t(((pixValidX * DDR_CH_IN +ciDiv * PE_KL_CI)/(PE_KL_CI * BRAM_1ADOTS))%MAX_WIDTH_X_CHIN_DIV4_KL);
                    //Write to pixChInColsKL, 1x11=11dots, one clock
                    L_LOOP_WRITE_TO_PIXCHINCOLSKL:
                    for(filter_t hKernel = 0; hKernel < KL_SIZE; hKernel++){
                    #pragma HLS UNROLL
                    	// y is begging from KL_SIZE-padSizeExtend,
                        // y - KL_SIZE = KL_SIZE-padSizeExtend-KL_SIZE = -2,
                    	// hKernel = 0,1,2,3,4,5,6,7,8,9,10
                        // y + hKernel - KL_SIZE = -2,-1,0,1,2,3,4,5,6,7,8
                        dimension_t pixValidY = y + hKernel - KL_SIZE;
                        bool isPadPixel = pixValidX < 0 | pixValidX >= WIDTH_IN | pixValidY < 0 | pixValidY >= HEIGHT_IN;
                        offset_t offsetY = isPadPixel ? offset_t(0) : offset_t(pixValidY % NUM_IMG_CACHE_LINES_KL);
                        offset_t offsetCiDiv = ciDiv % PE_KL_CI;
                        data_t pixTmp[BRAM_1ADOTS];
	                    #pragma HLS ARRAY_PARTITION variable = pixTmp complete dim = 0
//                        if( (0 == ((ciDiv * PE_KL_CI) % BRAM_1ADOTS)) && (!isPadPixel) ){
                        	ExtMemToApFixSync<DOT_BITS, apfix32_image_t,data_t,BRAM_1ADOTS >(IBRAM_KL[iAddr][offsetY][offsetCiDiv],pixTmp);
//                        	fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[0]));
//                        	fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[1]));
//                        	fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[2]));
//                        	fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[3]));
//                        }
                        for(offset_t ciDots =0; ciDots < PE_KL_CI; ciDots++){
                        #pragma HLS UNROLL
//                            pixChInColsKL[ciDiv][ciDots][hKernel] = isPadPixel ? data_t(0) : pixTmp[ciDots+((ciDiv*PE_KL_CI)%BRAM_1ADOTS)];
                        	pixChInColsKL[ciDiv][ciDots][hKernel] = isPadPixel ? data_t(0) : pixTmp[ciDiv];

//							fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[0]));
//							fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[1]));
//							fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[2]));
//							fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixTmp[3]));

//                        	fprintf(outFile,"fuck[%4d][%4d][%4d][%4d]=%4d\n",int(y),int(x),int(ciDiv),int(hKernel),int(pixChInColsKL[ciDiv][ciDots][hKernel]) );
//                          fprintf(outFile,"[%4d][%4d][%4d][%4d]=%4d\n",int(pixValidY),int(pixValidX),int(ciDiv),int(hKernel),int(pixChInColsKL[ciDiv][ciDots][hKernel]));
                        }
                    }
                    //pixChinColsKL --> bufferPixKL and winPixKL
                    // One Clock , insert 4x5 =20 dots from pixChInColsKL
                    // One Clock , generate 4x5x5 = 100 dots by sliding window for next computing 
                    for(offset_t ciDots = 0; ciDots < PE_KL_CI; ciDots++ ){
                    #pragma HLS UNROLL
                        if( 1 == chInPEDiv ){
                        // winPixKL No need
                            // One clock ,generate 121 x 1 = 121 dots for computing
                            for(filter_t hKernel = 0; hKernel < KL_SIZE; hKernel++){
                            #pragma HLS UNROLL
                                for(filter_t wKernel = 0; wKernel < KL_SIZE-1; wKernel++){
                                #pragma HLS UNROLL
                                    //// left shift
                                    bufferPixKL[ciDots][hKernel][wKernel] = bufferPixKL[ciDots][hKernel][wKernel+1];
                                }
                                bufferPixKL[ciDots][hKernel][KL_SIZE-1] = pixChInColsKL[ciDiv][ciDots][hKernel];
                            }
                        }else{
                            winPixKL[ciDiv][ciDots].shift_pixels_left();
                            winPixKL[ciDiv][ciDots].insert_right_col(pixChInColsKL[ciDiv][ciDots]);
                        }
                    }
                    //bufferPixKL and winPixKL --> px5StrmValue;
                    if( (x >= 0) && (x<widthSize) ){
                        for(offset_t ciDots = 0; ciDots < PE_KL_CI; ciDots++ ){
                        #pragma HLS UNROLL
                            for(filter_t hKernel = 0; hKernel <  KL_SIZE; hKernel++){
                            #pragma HLS UNROLL
//                            	fprintf(outFile,"[%4d][%4d][%4d][%4d]",int(y),int(x),int(ciDiv),int(hKernel));
                                for(filter_t wKernel = 0; wKernel <  KL_SIZE; wKernel++){
                                #pragma HLS UNROLL
                                    offset_t index = ciDots * KL_SIZE * KL_SIZE + hKernel * KL_SIZE + wKernel;
                                    data_t bufPix;
                                    if( 1 == chInPEDiv ){
                                        //remK = (KL_SIZE-KERNEL)/2;////remainder
                                        //denK = KERNEL+remK;//denominator
                                    	//For example, KL_SIZE = 5, KERNEL =3
                                    	//remK = 1, denK=3+1=4;
                                    	//hKernel,wKernel
                                    	//00,01,02,03,04
                                    	//10,11,12,13,14
                                    	//20,21,22,23,24
                                    	//30,31,32,33,34
                                    	//40,41,42,43,44
                                    	//We hope 11,12,13,21,22,23,31,32,33 hold, others are zero.
                                    	if( ((hKernel%denK)<remK) || ((wKernel%denK)<remK)){
                                    		bufPix = 0;//virtual padSize,set 0
                                    	}else{
                                    		bufPix = bufferPixKL[ciDots][hKernel][wKernel];
                                    	}
                                    }else{
                                    	if( ((hKernel%denK)<remK) || ((wKernel%denK)<remK)){
                                    		bufPix = 0;//virtual padSize,set 0
                                    	}else{
                                    		bufPix = winPixKL[ciDiv][ciDots].val[hKernel][wKernel];
                                    	}
                                    }
//                                    fprintf(outFile,"%4d ",int(bufPix));
                                    hls::AXISetBitFields<PXL_STRM_WIDTH, data_t>(px5StrmValue, index * DOT_BITS , DOT_BITS, bufPix);
                                }
//                                fprintf(outFile,"\n");
                            }
                        }
                        //px5StrmValue --> hlsPxlStrm;
                        if(!skipPixel){
                            hlsPxlStrm << px5StrmValue;
                        }
                    }//End if x >=0
                }//End if y >= (KL_SIZE-PAD_SIZE)
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
        hls_pxl_strm_t &hlsPxlStrm                                                   ,
        hls_outl_strm_t &hlsOutlStrm                                                 ,
        apfix32_weights_t WBRAM_KL[BLOCK_SIZE][PE_KL_CI][PE_KL_CO_PORT][SQR_KL_SIZE]
        ){
#pragma HLS inline off
    // write here by cqiu
    result_t OUTBRAM[MAX_NUM_SPLIT_CHOUT/PE_KL_CO][PE_KL_CO];
    #pragma HLS ARRAY_PARTITION variable = OUTBRAM complete dim = 2
    #pragma DEPENDENCE variable = OUTBRAM inter flase
    result_t OUTBRAM_BUF0[PE_KL_CO];
    #pragma HLS ARRAY_PARTITION variable = OUTBRAM_BUF0 complete dim = 0 
    pxl_strm_t pixelValue;
    data_t pixelsL[PE_KL_CI][SQR_KL_SIZE];
    #pragma HLS ARRAY_PARTITION variable = pixelsL complete dim = 0
    offset_t ciOffset;
    channels_t chInPEDiv = CEIL_DIV(CH_IN,PE_KL_CI);
    channels_t chOutPEDiv8 = CEIL_DIV(CH_OUT,PE_KL_CO);
    // 11x11 deal with 7x7,PAD_SIZE=0, so padSizeExtend = 2
    filter_t remK = (KL_SIZE-KERNEL)/2;////remainder
    dimension_t padSizeExtend = PAD_SIZE + dimension_t(remK);
    dimension_t widthSize = STRIDE*(1+(WIDTH_IN+2*padSizeExtend-KL_SIZE)/STRIDE);
    dimension_t heightSize = STRIDE*(1+(HEIGHT_IN+2*padSizeExtend-KL_SIZE)/STRIDE);
    L_COMPUTE_Y:
    for(dimension_t y = 0; y < KL_HEIGHT_STRIDE; y++){
    #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16 
        L_COMPUTE_X:
        for(dimension_t x = 0; x < widthSize; x++){
        #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16 
            L_COMPUTE_CIDIV4:
            for(channels_t ciDiv = 0; ciDiv < chInPEDiv ; ciDiv++){
            #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16 
                L_PARALLEL_CODIV8:
                for(channels_t coDiv8 = 0; coDiv8 < chOutPEDiv8; coDiv8++){
                #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32 avg = 16 
                #pragma HLS pipeline II = 1
                    bool skipPixel = (STRIDE == 2) & (x%2 | y%2);
                    if(!skipPixel){
                        //Load pixels5 
                        if(coDiv8 == 0){
                            hlsPxlStrm >> pixelValue;
                            //Load pixels5 
                            L_LOAD_PIXEL5_CI_PE:
                            for(channels_t ciDot = 0; ciDot < PE_KL_CI; ciDot++){
                            #pragma HLS unroll
                                L_LOAD_PIXEL_KL_PE:
                                for(channels_t k = 0; k < SQR_KL_SIZE; k++){
                                #pragma HLS unroll
                                    offset_t index = ciDot * SQR_KL_SIZE + k;
                                    hls::AXIGetBitFields<PXL_STRM_WIDTH,data_t>(pixelValue,index * DOT_BITS, DOT_BITS, pixelsL[ciDot][k]);
                                }//End k
								#ifndef __SYNTHESIS__
//								  if(CH_IN ==3){
////									  fprintf(outFile,"x,y,ciDiv=%4d,%4d,%4d\n",int(x),int(y),int(ciDiv));
//									  int m = 0;
//									  for(int i=0;i<KL_SIZE;i++){
//											m = i * KL_SIZE;
//											if(x/2 > 122){
//												fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d ] \n",
//													  (int)pixelsL[ciDot][m+0], (int)pixelsL[ciDot][m+1], (int)pixelsL[ciDot][m+2], (int)pixelsL[ciDot][m+3], (int)pixelsL[ciDot][m+4],
//													  (int)pixelsL[ciDot][m+5], (int)pixelsL[ciDot][m+6], (int)pixelsL[ciDot][m+7], (int)pixelsL[ciDot][m+8], (int)pixelsL[ciDot][m+9],
//													  (int)pixelsL[ciDot][m+10] );
//											}
//									  }
//								  }
								#endif
                            }//End ciDot
//                            ciOffset = ciDiv4 * CEIL_DIV(CH_OUT,BRAM_1ADOTS) * BRAM_1ADOTS;
                        }//End coDiv8 ==0
                        //Load weightsL
                        data_t weightsL[PE_KL_CO][PE_KL_CI][SQR_KL_SIZE];
                        #pragma HLS ARRAY_PARTITION variable = weightsL complete dim = 0
                        getKLWeights( coDiv8 * PE_KL_CO ,ciDiv * PE_KL_CI, WBRAM_KL,weightsL);
                        //Compute parallel
                        L_COMPUTE_CO_MUL:
                        for(channels_t coMul = 0; coMul < PE_KL_CO/2; coMul++ ){
                        #pragma HLS UNROLL
                            result_t result[2][PE_KL_CI];
							#pragma HLS ARRAY_PARTITION variable = result complete dim = 0
                            L_COMPUTE_CI_MUL:
                            for(channels_t ciMul = 0; ciMul < PE_KL_CI; ciMul++){
                            #pragma HLS UNROLL
                            	macc2dKL(pixelsL[ciMul],weightsL[2*coMul][ciMul],result[0][ciMul]);
                            	macc2dKL(pixelsL[ciMul],weightsL[channels_t(2*coMul+1)][ciMul],result[1][ciMul]);
//                            	macc2dKL_int8(pixelsL[ciMul],weightsL[2*coMul][ciMul],weightsL[channels_t(2*coMul+1)][ciMul],result[0][ciMul],result[1][ciMul] );
                            }
                            result_t resultSum[2];
							#pragma HLS ARRAY_PARTITION variable = resultSum complete dim = 0
                            L_COMPUTE_CO_MUL_DOUBLE:
                            for(channels_t coDouble = 0; coDouble < 2; coDouble++ ){
                            #pragma HLS UNROLL
//								if(CH_IN == 3){
//									resultSum[coDouble] = result[coDouble][0] + result[coDouble][1] + result[coDouble][2];
//								}else{
//									resultSum[coDouble] = result[coDouble][0] + result[coDouble][1] + result[coDouble][2] + result[coDouble][3];
//								}
                            	resultSum[coDouble] = result[coDouble][ciDiv%PE_KL_CI];

								if(CEIL_DIV(CH_OUT,PE_KL_CO) == 1){
									if(ciDiv == 0){
										OUTBRAM_BUF0[2*coMul+coDouble] = resultSum[coDouble];
									}else{
										OUTBRAM_BUF0[2*coMul+coDouble] = OUTBRAM_BUF0[2*coMul+coDouble] + resultSum[coDouble];
									}
								}else{
									if(ciDiv == 0){
										OUTBRAM[coDiv8][2*coMul+coDouble] = resultSum[coDouble];
									}else{
										OUTBRAM[coDiv8][2*coMul+coDouble] = OUTBRAM[coDiv8][2*coMul+coDouble] + resultSum[coDouble];
									}
								}
#ifndef __SYNTHESIS__
//									if(6 ==int( 2*coMul+coDouble + coDiv8 * 8)){
//										fprintf(outFile,"y%4d,x%4d,ci=%4d,d=%8f\n",int(y),int(x),int(ciDiv),
//												int(OUTBRAM[coDiv8][2*coMul+coDouble]));
//									}
#endif
                            }
                        }
                        //Sent Result 
                        if(ciDiv == chInPEDiv-1 ){
                            outl_strm_t outLStrmValue;
                            result_t outData[PE_KL_CO];
                            #pragma HLS ARRAY_PARTITION variable = outData complete dim = 0 
                            //Transmit the result to outData;
                            L_COMPUTE_POST:
                            for(channels_t coPost = 0 ; coPost < PE_KL_CO; coPost++){
                            #pragma HLS UNROLL
                                if(CEIL_DIV(CH_OUT,PE_KL_CO) == 1){
                                    outData[coPost] = OUTBRAM_BUF0[coPost];
                                }else{
                                    outData[coPost] = OUTBRAM[coDiv8][coPost];
                                }
                            }
                            L_COMPUTE_TRANSMIT_TO_STRM:
                            for(emem_1adots_t ememCnt = 0; ememCnt < PE_KL_CO; ememCnt++){
	                        #pragma HLS unroll

                            	if(outData[ememCnt] >= (1 << (2 * DOT_BITS-1)) ){
                            		outData[ememCnt] = (1 << (2 * DOT_BITS -1)) - 1 ;
                            	}

                            	if(outData[ememCnt] < (-(1 << (2 * DOT_BITS-1))) ){
                            		outData[ememCnt] = -(1 << (2 * DOT_BITS -1));
                            	}
                                hls::AXISetBitFields(outLStrmValue,int(ememCnt * DOT_BITS * 2), int(DOT_BITS *2) , data_result_t(outData[ememCnt]));
								#ifndef __SYNTHESIS__
//									fprintf(outFile,"y=%4d,x=%4d,coDiv8=%4d,outdata=%8f\n",
//											int(y),int(x),int(coDiv8*PE_KL_CO+ememCnt),float( (outData[ememCnt])/(128.0f)));
//									fprintf(outFile,"%8f\n",float((outData[ememCnt])/(128.0f)));
								#endif
                            }
                            hlsOutlStrm << outLStrmValue;
                        }// End Compute
                    }//End skipPixel
                }//End coDiv8
            }//End ciDiv4
        }//End x
    }//End y
}
//================================================================
//= getKLWeights
//================================================================
void getKLWeights(
        const channels_t coIndex                                                 ,
        const channels_t ciIndex                                                 ,
        apfix32_weights_t WBRAM[BLOCK_SIZE][PE_KL_CI][PE_KL_CO_PORT][SQR_KL_SIZE]   ,
        data_t weightsL[PE_KL_CO][PE_KL_CI][SQR_KL_SIZE]
        ){
    #pragma HLS FUNCTION_INSTANTIATE variable = coIndex
    #pragma HLS FUNCTION_INSTANTIATE variable = ciIndex
    #pragma HLS inline 
    #pragma HLS pipeline 
    //calculate address
    //ci0co0,ci0co1,ci0co2,ci0co3,ci1co0,ci1co2,ci1co3.......
    offset_t coAddrOffsetCiDiv = (ciIndex/PE_KL_CI) * CEIL_DIV(CH_OUT,PE_KL_CO)*PE_KL_CO;
    block_size_t blkAddr = ( (coIndex + coAddrOffsetCiDiv)/(BRAM_1ADOTS * PE_KL_CO_PORT) ) % BLOCK_SIZE;
//    data_t weightsTmpL[PE_KL_CO][PE_KL_CI][SQR_KL_SIZE];
//	#pragma HLS ARRAY_PARTITION variable = weightsTmpL complete dim = 0
    ////For example
    ////Such KL_SIZE = 5, KERNEL = 3
    //// 0  1  2  3  4
    //// 5  6  7  8  9
    ////10  11 12 13 14
    ////15  16 17 18 19
    ////20  21 22 23 24
    //// so firstPos = 6,
    filter_t fristPos = (KL_SIZE - KERNEL)*(KL_SIZE+1)/2;
//    filter_t posIndex = fristPos;
    L_GETWEIGHTS_CIDIV:
    for(channels_t ci = 0; ci < PE_KL_CI; ci++){
        L_GETWEIGHTS_KERNEL:
        for(filter_t k = 0; k < SQR_KL_SIZE; k++){
				L_GETWEIGHTS_CODIV4:
				for(channels_t coDiv4 = 0; coDiv4 < PE_KL_CO_PORT; coDiv4++){
					data_t weightsValue[BRAM_1ADOTS];
					ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>(WBRAM[blkAddr][ci][coDiv4][k],weightsValue);
#ifndef __SYNTHESIS__
//					if(1 == coDiv4){
//						fprintf(outFile,"[ %4d %4d %4d ]=%4d \n",int(ci),int(k),int(coDiv4),int(weightsValue[2]));
//					}
//    	  fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d] \n",
#endif
					// we only write the center valid weights.
#if 0
					if(k<KERNEL*KERNEL){
						weightsL[coDiv4 * BRAM_1ADOTS + 0][ci][fristPos+(k%KERNEL)+(k/KERNEL)*KL_SIZE] = weightsValue[0];
						weightsL[coDiv4 * BRAM_1ADOTS + 1][ci][fristPos+(k%KERNEL)+(k/KERNEL)*KL_SIZE] = weightsValue[1];
						weightsL[coDiv4 * BRAM_1ADOTS + 2][ci][fristPos+(k%KERNEL)+(k/KERNEL)*KL_SIZE] = weightsValue[2];
						weightsL[coDiv4 * BRAM_1ADOTS + 3][ci][fristPos+(k%KERNEL)+(k/KERNEL)*KL_SIZE] = weightsValue[3];
					}
#else
					weightsL[coDiv4 * BRAM_1ADOTS + 0][ci][k] = weightsValue[0];
					weightsL[coDiv4 * BRAM_1ADOTS + 1][ci][k] = weightsValue[1];
					weightsL[coDiv4 * BRAM_1ADOTS + 2][ci][k] = weightsValue[2];
					weightsL[coDiv4 * BRAM_1ADOTS + 3][ci][k] = weightsValue[3];
#endif


				}
        }
    }
}
//================================================================
//= macc2dKL_int8
//================================================================
void macc2dKLInt8Tmp(
        const data_t pixels5[SQR_KL_SIZE]   ,
        const data_t weights5Low[SQR_KL_SIZE]  , const data_t weights5High[SQR_KL_SIZE]  ,
         result_t &resultLow, result_t &resultHigh) {
#pragma HLS inline
	macc2dKL(pixels5,weights5Low,resultLow);
	macc2dKL(pixels5,weights5High,resultHigh);
}
//================================================================
//= macc2dKL_int8
//================================================================
void macc2dKL_int8(
        const data_t pixelsL[SQR_KL_SIZE]   ,
        const data_t weightsLLow[SQR_KL_SIZE]  , const data_t weightsLHigh[SQR_KL_SIZE]  ,
            result_t &resultLow, result_t &resultHigh ) {
#pragma HLS inline

#define  HIGH_SHIFT_BITS 18
    apfix45_dsp_out_t mulResult[SQR_KL_SIZE];
    apfix45_dsp_out_t accumulator[ACCU_GROUP] = { 0, 0, 0, 0 };
    #pragma HLS ARRAY_PARTITION variable = mulResult complete dim = 0

    apfix27_weight_t weightsL_comb[SQR_KL_SIZE];

    result_t accuResultLow[ACCU_GROUP];
    result_t accuResultHigh[ACCU_GROUP];

    int pixelsL_int[SQR_KL_SIZE];

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
      for (int i = 0; i < SQR_KL_SIZE; i++) {
    #pragma HLS UNROLL
    	  weightsL_comb[i] = (( apfix27_weight_t( weightsLHigh[i] ))<<HIGH_SHIFT_BITS) +apfix27_weight_t( weightsLLow[i]);
      }

    L_MACC_MULTIPLY:
    for(filter_t i = 0; i< SQR_KL_SIZE; i++){
    #pragma HLS UNROLL
        mulResult[i] = pixelsL[i] * weightsL_comb[i] ;
    }

     L_MACC_ACCUMULATE:
	for( accu_group_t group=0; group<ACCU_GROUP; group++ ){
#pragma HLS UNROLL
        for(int i = group*INT8_TIMES; i< ( group+1)*INT8_TIMES; i++){
        	if( i<SQR_KL_SIZE ){
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
void macc2dKL(
        const data_t pixels5[SQR_KL_SIZE]   ,
        const data_t weights5[SQR_KL_SIZE]  ,
            result_t &result ) {
#pragma HLS inline
    result_t accumulator = 0;
    data_result_t mulResult[SQR_KL_SIZE];
    #pragma HLS ARRAY_PARTITION variable = mulResult complete dim = 0

    #ifndef __SYNTHESIS__
#if 1
      if(CH_IN ==3){
//          int m = 0;
//          for(int i=0;i<KL_SIZE;i++){
//                m = i * KL_SIZE;
//    	    fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d ] \n",
//    			  (int)pixels5[m+0], (int)pixels5[m+1], (int)pixels5[m+2], (int)pixels5[m+3], (int)pixels5[m+4],
//    			  (int)pixels5[m+5], (int)pixels5[m+6], (int)pixels5[m+7], (int)pixels5[m+8], (int)pixels5[m+9],
//    			  (int)pixels5[m+10] );
//          }
//          for(int i=0;i<KL_SIZE;i++){
//                m = i * KL_SIZE;
//             fprintf(outFile,"Weights:[ %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d ] \n",
//  			  (int)weights5[m+0], (int)weights5[m+1], (int)weights5[m+2], (int)weights5[m+3], (int)weights5[m+4],
//  			  (int)weights5[m+5], (int)weights5[m+6], (int)weights5[m+7], (int)weights5[m+8], (int)weights5[m+9],
//  			  (int)weights5[m+10] );
//          }
      }
#else
      if(CH_IN ==3){
//          fprintf(outFile,"Pixels:\n");
//          for(int i=0;i<KL_SIZE;i++){
//        	  for(int j=0;j<KL_SIZE;j++){
//        		  fprintf(outFile,"%4d \n",(int)pixels5[i*KL_SIZE+j]);
//        	  }
//          }
//			fprintf(outFile,"Weights:\n");
//			for(int i=0;i<KL_SIZE;i++){
//			  for(int j=0;j<KL_SIZE;j++){
//				  fprintf(outFile,"%4d \n",(int)weights5[i*KL_SIZE+j]);
//			  }
//			}
      }
//                m = i * KL_SIZE;
//    	    fprintf(outFile,"Pixels: [ %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d ] \n",
//    			  (int)pixels5[m+0], (int)pixels5[m+1], (int)pixels5[m+2], (int)pixels5[m+3], (int)pixels5[m+4],
//    			  (int)pixels5[m+5], (int)pixels5[m+6], (int)pixels5[m+7], (int)pixels5[m+8], (int)pixels5[m+9],
//    			  (int)pixels5[m+10] );
//          }
//          for(int i=0;i<KL_SIZE;i++){
//                m = i * KL_SIZE;
//             fprintf(outFile,"Weights:[ %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d ] \n",
//  			  (int)weights5[m+0], (int)weights5[m+1], (int)weights5[m+2], (int)weights5[m+3], (int)weights5[m+4],
//  			  (int)weights5[m+5], (int)weights5[m+6], (int)weights5[m+7], (int)weights5[m+8], (int)weights5[m+9],
//  			  (int)weights5[m+10] );
//          }
//      }
#endif
    #endif
#if 1
    L_MACC_MULTIPLY:
    for(filter_t i = 0; i< SQR_KL_SIZE; i++){
    #pragma HLS UNROLL
        mulResult[i] = pixels5[i] * weights5[i] ;
    }
    L_MACC_ACCUMULATE:
    for(filter_t i = 0; i< SQR_KL_SIZE; i++){
    #pragma HLS UNROLL
       accumulator = accumulator + result_t(mulResult[i]);
    }
    result = accumulator ;
#else
    result = 0;
#endif

//    for( int i=0; i<SQR_KL_SIZE; i++ ){
//    	printf("i=%d; pixels5=%d\n",i, int(pixels5[i]) );
//    }
//
//    for( int i=0; i<SQR_KL_SIZE; i++ ){
//    	printf("i=%d; weights5=%d\n",i, int(weights5[i]) );
//    }
//
//
//   printf("result=%d\n",int(result) );
//    fprintf(outFile,"result = %4d\n",int(result));
}

//================================================================
//= writeBackToDRAM
//================================================================
void writeBackToDRAM(
        hls_outl_strm_t &hlsOutlStrm,
        iemem_t *WRITE_SHARED_DRAM,
        apfix32_weights_t BBRAM_KL[BLOCK_SIZE][BIAS_NUM][WEMEM_1ADOTS_DIV8][PE_KL_CO_PORT]
    ){
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable = LBRAM complete dim = 2 // EXTMEM_ALIGNMENT
#pragma HLS RESOURCE variable = LBRAM core = RAM_T2P_BRAM latency = 3
//data_t LBRAM[MAX_WIDTH_X_CHIN_DIV16][WEMEM_1ADOTS];////512x16
data_t maxData = 0;
data_t minData = 255;
//dimension_t widthOut = (STRIDE ==2) ? dimension_t(WIDTH_IN/2) : WIDTH_IN ;
num_sublayers_t subLayerMulti = (SUB_LAYER_FLAG == 0) ? num_sublayers_t(1) : SUB_LAYER_NUM   ;
offset_t xOffsetDiv16 = subLayerMulti * CEIL_DIV(CH_OUT,IEMEM_1ADOTS);
offset_t yOffsetDiv16 = xOffsetDiv16 * KL_WIDTH ;
result_t halfValueCaffe = 1<<(IMG_POS_IN + WEIGHTS_POS - IMG_POS_OUT-1);
result_t halfValueYolo  = 1<<(IMG_POS_IN + WEIGHTS_POS + SCALES_POS - IMG_POS_OUT-1);

    L_WRITEBACK_Y:
    for(dimension_t y = 0 ; y < KL_HEIGHT + 1; y++){
    #pragma HLS LOOP_TRIPCOUNT MIN = 8 AVG = 16 MAX = 32 
        L_WRITEBACK_X:
        for(dimension_t x = 0; x < KL_WIDTH; x++){
        #pragma HLS LOOP_TRIPCOUNT MIN = 8 AVG = 16 MAX = 32 
            L_WRITEBACK_CODIV16:
            for(channels_t coDiv8 = 0; coDiv8 < CEIL_DIV(CH_OUT,PE_KL_CO); coDiv8++){
            #pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 2 MAX = 4 
            #pragma HLS pipeline II = 1
                data_result_t raw[PE_KL_CO];
                result_t bias[PE_KL_CO];
                result_t mean[PE_KL_CO];
                result_t variance[PE_KL_CO];
                result_t batchData[PE_KL_CO];
                result_t batchDataRelu[PE_KL_CO];
                result_t batchDataDiv10Int[PE_KL_CO];
                data_t activeData[PE_KL_CO];
                #pragma HLS ARRAY_PARTITION variable = raw complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = variance complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = mean complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = bias complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = batchData complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = batchDataRelu complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = batchDataDiv10Int complete dim = 0 
                #pragma HLS ARRAY_PARTITION variable = activeData complete dim = 0 
                data_t preX[MAX_CHANNELS/PE_KL_CO][PE_KL_CO];
                data_t curX[MAX_CHANNELS/PE_KL_CO][PE_KL_CO];
                data_t maxX[MAX_CHANNELS/PE_KL_CO][PE_KL_CO];
                data_t tmpX[MAX_CHANNELS/PE_KL_CO][PE_KL_CO];
                #pragma HLS ARRAY_PARTITION variable = preX complete dim = 2
                #pragma HLS ARRAY_PARTITION variable = curX complete dim = 2
                #pragma HLS ARRAY_PARTITION variable = maxX complete dim = 2
                #pragma HLS ARRAY_PARTITION variable = tmpX complete dim = 2
                memaddr_t ddrAddr ;
                half_iemem_t halfIememTmp;
                iemem_t iememTmp;


                //The last row process
                if( y == KL_HEIGHT ){
                    if(POOLING_TYPE == 1){// Max Pooling Stride == 1 
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + (y-1) * yOffsetDiv16 + x * xOffsetDiv16 + coDiv8/2;  
                        ApFixToExtMemSync<DOT_BITS,half_iemem_t,data_t,PE_KL_CO>(halfIememTmp,LBRAM[offset_t(x * xOffsetDiv16 *2 + coDiv8)]);
                    }
                }else{
                //Normal rows process
                    //Load raws values 
                    outl_strm_t outResult;
                    hlsOutlStrm >> outResult;
                    ExtMemToApFixSync<DOT_BITS * 2 ,outl_strm_t,data_result_t,PE_KL_CO>(outResult,raw);
                    data_t apfixTmp[3][PE_KL_CO_PORT][BRAM_1ADOTS];
                    //Load bias to apfixTmp[0]
                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
                        (BBRAM_KL[coDiv8/2][0][coDiv8%2][0],apfixTmp[0][0]);
                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
                        (BBRAM_KL[coDiv8/2][0][coDiv8%2][1],apfixTmp[0][1]);
//                    //Load means to apfixTmp[1]
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_KL[coDiv8/2][2][coDiv8%2][0],apfixTmp[1][0]);
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_KL[coDiv8/2][2][coDiv8%2][1],apfixTmp[1][1]);
//                    //Load variance to apfixTmp[2]
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_KL[coDiv8/2][3][coDiv8%2][0],apfixTmp[2][0]);
//                    ExtMemToApFixSync<DOT_BITS,apfix32_weights_t,data_t,BRAM_1ADOTS>
//                        (BBRAM_KL[coDiv8/2][3][coDiv8%2][1],apfixTmp[2][1]);
                    for(channels_t coIndex = 0; coIndex < PE_KL_CO; coIndex++){
                    #pragma HLS UNROLL factor = PE_KL_CO
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
                            activeData[coIndex] = data_result_t( (batchDataRelu[coIndex]+halfValueYolo) >> (IMG_POS_IN + WEIGHTS_POS + SCALES_POS - IMG_POS_OUT));
                        }else{

                            batchData[coIndex] = raw[coIndex] + (bias[coIndex] << (IMG_POS_IN + WEIGHTS_POS - BIAS_POS)); 
                            batchDataRelu[coIndex] = (batchData[coIndex] > 0) ? batchData[coIndex] : result_t(0) ;
//                            batchDataRelu[coIndex] =  batchData[coIndex] ;
                            activeData[coIndex] = data_result_t( (batchDataRelu[coIndex]+halfValueCaffe ) >> (IMG_POS_IN + WEIGHTS_POS - IMG_POS_OUT));
							#ifndef __SYNTHESIS__
//                            		int valTimes=1;
//                            	if(IMG_POS_OUT < 0){
//                            		valTimes = 1 << (-IMG_POS_OUT);
//                            		fprintf(outFile,"%8f\n",float(activeData[coIndex]*1.0f * valTimes ));
//                            	}else{
//                            		valTimes = 1 << (IMG_POS_OUT);
//                            		fprintf(outFile,"%8f\n",float(activeData[coIndex]*1.0f / valTimes ));
//                            	}
							#endif
                        }
                        if(x == 0){
                            preX[coDiv8][coIndex] = -MAX_SHORT;
                        } else {
                            preX[coDiv8][coIndex] = curX[coDiv8][coIndex]; 
                        }
                        if(POOLING_TYPE == 2){// Max Pooling Stride == 2
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


                    if(POOLING_TYPE == 2){// Max Pooling Stride == 2 
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + (y/2) * yOffsetDiv16 / 2 + (x/2) * xOffsetDiv16 + coDiv8/2;  
                        ApFixToExtMemSync<DOT_BITS,half_iemem_t,data_t,PE_KL_CO>(halfIememTmp,maxX[coDiv8]);
                    }else if(POOLING_TYPE == 0){// No Max Pooling
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + y * yOffsetDiv16  + x * xOffsetDiv16 + coDiv8/2;  
                        ApFixToExtMemSync<DOT_BITS,half_iemem_t,data_t,PE_KL_CO>(halfIememTmp,activeData);

                    }else if(POOLING_TYPE == 1){// Max Pooling Stride == 1 
                        ddrAddr = IMAGE_OFFSET + LAYER_OUT_IMAGE_OFFSET + (y-1) * yOffsetDiv16  + x * xOffsetDiv16 + coDiv8/2;  
                        ApFixToExtMemSync<DOT_BITS,half_iemem_t,data_t,PE_KL_CO>(halfIememTmp,maxX[coDiv8]);
                    }
                }//End Normal rows
                if( (   (y != KL_HEIGHT) &&
                        (((POOLING_TYPE == 2) && (y%2) && (x%2)) || (POOLING_TYPE == 0)) )
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
                    		int valTimes=1;
                    		//// conv1 check ok
							if(IMG_POS_OUT < 0){
								valTimes = 1 << (-IMG_POS_OUT);
								fprintf(outFile,"%4d\n",int(resultValue[rCnt] * valTimes ));
							}else{
								valTimes = 1 << (IMG_POS_OUT);
								fprintf(outFile,"%4d\n",int(resultValue[rCnt] / valTimes ));
							}
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
