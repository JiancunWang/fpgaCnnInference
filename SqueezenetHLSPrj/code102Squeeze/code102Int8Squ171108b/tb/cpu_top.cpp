//----------------------------------------------------------------
//  FPGA Accelerator For CNN Inference
//----------------------------------------------------------------
//
//  File:   cpu_top.cpp
//  CPU-Side Functions for FPGA Accelerator
//
//  (c) qiu chao, 2017-09
//
//----------------------------------------------------------------

#include "cpu_top.hpp"
#include <math.h>
//#include <cblas.h>

//================================================================
//=Global Variables
//================================================================
layer_weights_t *SHARED_DRAM;
wemem_t *SHARED_DRAM_WEIGHTS;
iemem_t *SHARED_DRAM_IMAGE;
//================================================================

//=Main
//================================================================
int main(){
    printf(" Allocate Memory Regions for Data + Weights                           \n");        
    printf(" Reserve Memory and Assign Pointers                                   \n");
    printf(" DRAM:                                                                \n");
    printf("      ____________________                                            \n");
    printf("     | SHARED_DRAM_WEIGHTS|  0                                        \n");
    printf("     |              	  |  ...                                      \n");
    printf("     |____________________|  weightsSizeBytes - 1                     \n");
    printf("     | SHARD_DRAM_IMAGE   |  weightsSizeBytes                         \n");
    printf("     |   in + output 	  |  ...                                      \n");
    printf("     |____________________|  weightsSizeBytes + imageSizeBytes- 1     \n");
    // Setup NetWork Memory
    network_t *netCpu;
    netCpu = getNetWorkConfig();
    // Setup FPGA SHARED DRAM Memory
    long weightsSizeBytes = std::ceil(netCpu->numWeightsBytes * 1.0f / SIZEOF_IEMEM )*SIZEOF_IEMEM;
    long imageSizeBytes = std::ceil(netCpu->numImageBytes * 1.0f / SIZEOF_IEMEM ) * SIZEOF_IEMEM;
    long totalSize = weightsSizeBytes + imageSizeBytes; 
    SHARED_DRAM         = (layer_weights_t *)malloc(totalSize);
    SHARED_DRAM_WEIGHTS = (wemem_t *)(SHARED_DRAM);
    SHARED_DRAM_IMAGE   = (iemem_t *)(SHARED_DRAM + weightsSizeBytes);
    printf("totalSize = %8u\n",totalSize);
    if(DRAM_DEPTH != totalSize/SIZEOF_IEMEM){
    	printf("\n\n DRAM_DEPTH Set ERROR !! \n\n");
    	printf("Please Set DRAM_DEPTH = %8u\n",totalSize/SIZEOF_IEMEM);
    	exit(-1);
    }
    printf("Weights transfer datapath...............                                \n");  
    printf("           (malloc)          (transform)        (memcpy)                \n");
    printf("WeightsFile --> weightsCpuAddr --> weightsBaseAddr --> SHARED_DRAM_WEIGHTS\n");  
    printf("image   transfer datapath...............                                \n");  
    printf("           (malloc)          (transform)        (memcpy)                \n");
    printf("imageFile   --> imgInCpuAddr  -->  imgInBaseAddr --> SHARED_DRAM_IMAGE  \n");  
    // basePath = ---/csim/build
    const char* fileName = "../../../../../../dataSqueeze/weightsSqu.w";
//    const char* fileName = "cFarWeights.w";
    loadWeightsFromFile(netCpu,fileName);
    // Copy Weights to SHARED_DRAM_WEIGHTS
    memcpy(SHARED_DRAM_WEIGHTS,netCpu->weightsBaseAddr,netCpu->numWeightsBytes);
    // load Images 
    layer_t *layer0 = &netCpu->layers[0];
    cpu_data_t *imgInCpuAddr = (cpu_data_t *)malloc(sizeof(cpu_data_t) * layer0->width * layer0->height * layer0->channelsIn );
     const char* imgFileName = "../../../../../../dataSqueeze/catSqu.i";
//    const char* imgFileName = "birdCFar.i";
    loadInputImage(imgFileName,imgInCpuAddr,layer0);
    // transform Images 
    int widthIn     = layer0->width;
    int heightIn    = layer0->height;
    int channelIn   = layer0->channelsIn;
    int multiTimes  = 1 << (layer0->imgPosIn);
    // real image size = widthIn * heightIn * 3
    // We chanege the size to = widthIn * heightIn * IEMEM_1ADOTS;
    int inImgSize   = widthIn * heightIn * IEMEM_1ADOTS;
    layer_img_t *imgInBaseAddr = (layer_img_t *)malloc(inImgSize * DOT_BYTES);
    transformInputImage(multiTimes,inImgSize,imgInCpuAddr,imgInBaseAddr);
    // Copy image to SHARED_DRAM_IMAGE
    memcpy(SHARED_DRAM_IMAGE,imgInBaseAddr,inImgSize * DOT_BYTES);
    // main loop body
    offset_t weightsOffset = ((long)SHARED_DRAM_WEIGHTS - (long)SHARED_DRAM)/SIZEOF_IEMEM;
    offset_t imageOffset  = ((long)SHARED_DRAM_IMAGE - (long)SHARED_DRAM)/SIZEOF_IEMEM;
    weights_t nextWeightsLayer;
//    for (int layerId = 0; layerId < netCpu->numLayers; layerId++){
    for (int layerId = 0; layerId < 2; layerId++){
        layer_t layer[2];
        layer[0] = netCpu->layers[layerId];
        layer[1] = netCpu->layers[layerId];
        nextWeightsLayer.layerNum = layerId;
        nextWeightsLayer.channelsIn = layer[1].channelsIn;
        nextWeightsLayer.channelsOut = layer[1].channelsOut;
        nextWeightsLayer.kernel = layer[1].kernel;
        // nextWeightsLayer.weightsNum = 1;
        nextWeightsLayer.memAddrWeights = layer[1].memAddrWeights/sizeof(wemem_t);
        nextWeightsLayer.layerBiasOffset[0] = 0 ;
        nextWeightsLayer.layerBiasOffset[1] = 0 ;
        nextWeightsLayer.layerBiasOffset[2] = 0 ;
        nextWeightsLayer.layerBiasOffset[3] = 0 ;
        printf("FPGA Test Begin\n");
        fpga_top(
                (wemem_t *)SHARED_DRAM,
                (iemem_t *)SHARED_DRAM,
                (iemem_t *)SHARED_DRAM,
        		layer[0],
				nextWeightsLayer,
				weightsOffset,
				imageOffset
				);
    }

//    //InnerProjuct Process
//    layer_t lastLayer = netCpu->layers[netCpu->numLayers-1];
//	int ch0 = 1024;
//	int ch1 = lastLayer.channelsOut;//64
//	int ch2 = 10;
//	memaddr_t resultOffset = imageOffset * sizeof(iemem_t) + lastLayer.memAddrImgOut ;
//	data_t *iBuffData = (data_t *)malloc(sizeof(data_t) * ch0);
//	data_t *iBuffDataCp1 = iBuffData;
//	memcpy(iBuffData,( (data_t *)SHARED_DRAM+resultOffset) , ch0*sizeof(data_t));
//	cpu_data_t *iBuffCpuData = (cpu_data_t *)malloc(sizeof(cpu_data_t) * ch0);
//	cpu_data_t *iBuffCpuDataCp1 = iBuffCpuData;
//	for(int iCnt = 0 ; iCnt < ch0 ; iCnt++){
//		*iBuffCpuDataCp1++ = (*iBuffDataCp1++)*1.0f/(1 << lastLayer.imgPosOut);
////		printf("iBuffResult[%4d]=%12f\n",iCnt,*(iBuffCpuDataCp1-1));
//	}
//	cpu_data_t *rBuff = (cpu_data_t *)malloc(sizeof(cpu_data_t) * ch2);
//    innerProductProcess(iBuffCpuData,rBuff,ch0,ch1,ch2);
//    softMax(rBuff,ch2);
//    free(iBuffData);
//    free(rBuff);
    free(SHARED_DRAM);
    free(imgInCpuAddr);
    free(imgInBaseAddr);
    return 0;
}



//================================================================
//= softMax
//================================================================
void softMax(
		cpu_data_t *rBuff,
		int chIn
){
	//initial labels
	char labels[10][32]={
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck"};
	softMaxResult_t softMaxResultArray[10];
	for(int i = 0 ; i < chIn ; i++ ){
		softMaxResultArray[i].setName(labels[i]);
		softMaxResultArray[i].setValue(0.0f);
	}
	cpu_data_t *rBuffCp1 = rBuff;
	cpu_data_t *dataBuff = (cpu_data_t *)malloc(sizeof(cpu_data_t)*chIn);
	cpu_data_t *dataBuffCp1 = dataBuff;
	cpu_data_t maxData = -16384;
	cpu_data_t sumData = 0;
	//Find Max Data
	for(int ch = 0 ; ch < chIn ; ch++){
		if(maxData < *rBuffCp1){
			maxData = *rBuffCp1;
		}
		rBuffCp1++;
	}
	rBuffCp1 = rBuff;
	//Subtraction , EXP , Sum
	for(int ch = 0 ; ch < chIn ; ch++){
		*dataBuffCp1 = exp( (*rBuffCp1++) - maxData);
		sumData += *dataBuffCp1;
		dataBuffCp1++;
	}
	dataBuffCp1 = dataBuff;
	//Div
	for(int ch = 0 ; ch < chIn ; ch++){
		softMaxResultArray[ch].value = *dataBuffCp1/sumData;
		dataBuffCp1++;
	}
	//bubble_sort
	softMaxResult_t softMaxResultTmp;
	for(int i = 0 ; i < chIn-1 ; i++){
		for(int j = 0 ; j < chIn-1 -i; j++ ){
			if(softMaxResultArray[j].value < softMaxResultArray[j+1].value ){
				softMaxResultTmp.setName(softMaxResultArray[j].name);
				softMaxResultTmp.setValue(softMaxResultArray[j].value);
				softMaxResultArray[j].setName(softMaxResultArray[j+1].name);
				softMaxResultArray[j].setValue(softMaxResultArray[j+1].value);
				softMaxResultArray[j+1].setName(softMaxResultTmp.name);
				softMaxResultArray[j+1].setValue(softMaxResultTmp.value);
			}
		}
	}
	//Display
	for(int ch = 0 ; ch < chIn ; ch++){
		printf("[%12f - %12s]\n",softMaxResultArray[ch].value,softMaxResultArray[ch].name);
	}

	free(dataBuff);
}
//================================================================
//= innerProductProcess
//================================================================
void innerProductProcess(
		cpu_data_t *iBuff,
		cpu_data_t *rBuff,
		int ch0 ,
		int ch1 ,
		int ch2
){
//	const char* i0FileName = "../../../../../../dataSqueeze/innerProductImages0.w";
//	cpu_data_t *iBuff = (cpu_data_t *)malloc(sizeof(cpu_data_t) * i0Num);
//	loadInputFile(i0FileName,iBuff,ch0);
//	const char* i1FileName = "../../../../../../dataSqueeze/innerProductImages1.w";

	const char* w0FileName = "../../../../../../dataSqueeze/innerProductWeights0.w";
	const char* b0FileName = "../../../../../../dataSqueeze/innerProductBias0.w";
	const char* w1FileName = "../../../../../../dataSqueeze/innerProductWeights1.w";
	const char* b1FileName = "../../../../../../dataSqueeze/innerProductBias1.w";

//	const char* w0FileName = "innerProductWeights0.w";
//	const char* b0FileName = "innerProductBias0.w";
//	const char* w1FileName = "innerProductWeights1.w";
//	const char* b1FileName = "innerProductBias1.w";
	//load i0Buff
	int i0Num = ch0;
	cpu_data_t *i0Buff = (cpu_data_t *)malloc(sizeof(cpu_data_t) * i0Num);
	hwcTransTochw(iBuff,i0Buff,1,ch0/ch1,ch1);
	//load w0Buff
	int w0Num = ch0*ch1;
	cpu_data_t *w0Buff = (cpu_data_t *)malloc(sizeof(cpu_data_t) * w0Num);
	loadInputFile(w0FileName,w0Buff,w0Num);
	//load b0Buff
	int b0Num = ch1;
	cpu_data_t *b0Buff = (cpu_data_t *)malloc(sizeof(cpu_data_t) * b0Num);
	loadInputFile(b0FileName,b0Buff,b0Num);
	//new r0Buff
	cpu_data_t *r0Buff = (cpu_data_t *)malloc(sizeof(cpu_data_t) * b0Num);
	//calculate r0Buff
	arrayMulAdd(i0Buff,w0Buff,b0Buff,r0Buff,0,ch0,ch1);
	//load w1Buff
	int w1Num = ch2*ch1;
	cpu_data_t *w1Buff = (cpu_data_t *)malloc(sizeof(cpu_data_t) * w1Num);
	loadInputFile(w1FileName,w1Buff,w1Num);
	//load b1Buff
	int b1Num = ch2;
	cpu_data_t *b1Buff = (cpu_data_t *)malloc(sizeof(cpu_data_t) * b1Num);
	loadInputFile(b1FileName,b1Buff,b1Num);
	arrayMulAdd(r0Buff,w1Buff,b1Buff,rBuff,1,ch1,ch2);
	free(i0Buff);
	free(w0Buff);
	free(b0Buff);
	free(r0Buff);
	free(w1Buff);
	free(b1Buff);
}
//================================================================
//= hwcTransTochw
//================================================================
void hwcTransTochw(
		cpu_data_t *w0BuffIn,
		cpu_data_t *w0BuffOut,
		bool enIn,
		int lenIn,
		int chIn
	){
	for(int ci = 0 ; ci < chIn ; ci++){
		for(int len = 0 ; len < lenIn ; len++){
			if(enIn){
				*w0BuffOut++ = *(w0BuffIn + ci + len * chIn);
			}else{
				*w0BuffOut++ = *(w0BuffIn + ci*lenIn + len);
			}
		}
	}
}
//================================================================
//= arrayMulAdd
//================================================================
void arrayMulAdd(
		cpu_data_t *iBuff,
		cpu_data_t *wBuff,
		cpu_data_t *bBuff,
		cpu_data_t *rBuff,
		bool logEn,
		int chIn,
		int chOut){
	for(int co = 0 ; co < chOut ; co++){
		cpu_data_t *iBuffTmp = iBuff;
		cpu_data_t tmpSum = 0;
		for(int ci = 0 ; ci < chIn ; ci++){
			tmpSum += (*iBuffTmp++) * *wBuff++;
		}
		tmpSum = tmpSum + (*bBuff++);
		if(logEn){
//			printf("tmpSumAfterBias[%4d]= %12f\n",co,tmpSum);
			printf("%12f\n",tmpSum);
		}
		*rBuff++ = tmpSum;
	}
}
//================================================================
//= innerProductProcess
//================================================================
void loadInputFile(
        const char *fileName,
        cpu_data_t *inCpuAddr,
		int num){
    FILE *inFile = fopen(fileName,"rb");
    if(!inFile){
        printf("ERROR: File %s could not be opened!\n",fileName);
        exit(-1);
    }
    for(int ch =0; ch < num; ch++){
    	fscanf(inFile, "%f", inCpuAddr++);
    }
    fclose(inFile);
}
//================================================================
//= transformInputImage
//================================================================
void transformInputImage(
    int multiTimes                  ,
    int inImgSize                   ,
    cpu_data_t  *imgInCpuAddr       ,
    layer_img_t *imgInBaseAddr        
    ){
    cpu_data_t *imgInCpuAddrCnt = imgInCpuAddr;
    uatom_t *apfixIp = (uatom_t *)imgInBaseAddr;
    for(int i = 0; i < inImgSize; i = i + IEMEM_1ADOTS){
        for(int ci = 0; ci < IEMEM_1ADOTS; ci++){
            if(ci < 3){
                atom_t dataTmp = (*(imgInCpuAddrCnt++)) * multiTimes ;
                *apfixIp++ = *(uatom_t *)&dataTmp;
//                printf("InputImage[%4d][%4d]=% 4d\n",int(i/IEMEM_1ADOTS),ci,dataTmp);
            }else{
                *apfixIp++ = 0; 
            }
        }
    }
}
//================================================================
//= loadInputImage
//================================================================
void loadInputImage(
        const char *fileName,
        cpu_data_t *imgInCpuAddr,
        layer_t *layer){
    printf("===============================================================\n");  
    printf("===============================================================\n");  
	printf(" Caffe Image Formate----||          FPGA CNN Image Formate-----\n");
	printf(" Source Image --------  ||          Destinate Image -------    \n");
    printf(" ch means image input channels --------------------------------\n");
    printf(" h  means image height-----------------------------------------\n");
    printf(" w  means image width -----------------------------------------\n");
	printf(" ********************** ||          ************************   \n");
	printf(" ch0,h0,w0,w1,w2,w3..wn ||          h0,w0,ch0,ch1,ch2          \n");
	printf(" ch0,h1,w0,w1,w2,w3..wn ||          h0,w1,ch0,ch1,ch2          \n");
	printf(" ch0,h2,w0,w1,w2,w3..wn ||          h0,w2,ch0,ch1,ch2          \n");
	printf(" ...................... ||          ......................     \n");
	printf(" ch0,hn,w0,w1,w2,w3..wn ||          h0,wn,ch0,ch1,ch2          \n");
	printf(" ---------------------- ||     \\\\   h1,w0,ch0,ch1,ch2          \n");
	printf(" ch1,h0,w0,w1,w2,w3..wn || =====\\\\  h1,w1,ch0,ch1,ch2          \n");
	printf(" ch1,h1,w0,w1,w2,w3..wn || =====//  h1,w2,ch0,ch1,ch2          \n");
	printf(" ch1,h2,w0,w1,w2,w3..wn ||     //   ......................     \n");
	printf(" ...................... ||          h1,wn,ch0,ch1,ch2          \n");
	printf(" ch1,hn,w0,w1,w2,w3..wn ||          ----------------------     \n");
	printf(" ---------------------- ||          hn,w0,ch0,ch1,ch2          \n");
	printf(" ch2,h0,w0,w1,w2,w3..wn ||          hn,w1,ch0,ch1,ch2          \n");
	printf(" ch2,h1,w0,w1,w2,w3..wn ||          hn,w2,ch0,ch1,ch2          \n");
	printf(" ch2,h2,w0,w1,w2,w3..wn ||          ......................     \n");
	printf(" ...................... ||          hn,wn,ch0,ch1,ch2          \n");
	printf(" ch2,hn,w0,w1,w2,w3..wn ||          **********************     \n");
	printf(" ********************** ||          **********************     \n");
//    int numPixels = layer->width * layer->height * layer->channelsIn;
    FILE *inFile = fopen(fileName,"rb");
    if(!inFile){
        printf("ERROR: File %s could not be opened!\n",fileName);
        exit(-1);
    }
#if 0 // Caffe Image Formate
    for(int ch =0; ch < layer->channelsIn; ch++){
    	for(int h = 0; h < layer->height; h++){
    		for(int w = 0; w < layer->width; w++){
                fscanf(inFile, "%f", 
                        imgInCpuAddr + ch + w * layer->channelsIn + h * layer->channelsIn * layer->height );
            }
        }
    }
#else // FPGA CNN Image Formate
    for(int h = 0; h < layer->height; h++){
    	for(int w = 0; w < layer->width; w++){
    		for(int ch =0; ch < layer->channelsIn; ch++){
                fscanf(inFile, "%f",
                		imgInCpuAddr + ch + w * layer->channelsIn + h * layer->channelsIn * layer->width );
            }
        }
    }
#endif
    fclose(inFile);
}
//================================================================
//= fileOpen and fileClose
//================================================================
    FILE *outFileCpu;
    void fileOpenCpu(layer_t &layer) {
        char fileName[50];
        char layerName[10];
        int i = 0;
        while (char c = layer.name[i]){
            layerName[i++] = (c == '/' | c == ' ') ? '_' : c;
        }
        layerName[i] = '\0';
        sprintf(fileName, "cpuSimulation_%s.bin", layerName);
        outFileCpu = fopen(fileName, "w+");
    }
    void fileCloseCpu(){
        fclose(outFileCpu);
    }
