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
#include "network.hpp"
//================================================================
//=Config netWork
//================================================================
network_t *getNetWorkConfig(){
    network_t *net = new network_t(TOTAL_NUM_LAYERS,TOTAL_WEIGHT_BYTES);
    ////  const float MEAN_R = 104;      ( 0 - 255 ) - 104 =  (-104,151)
    ////  const float MEAN_G = 117;      ( 0 - 255 ) - 117 =  (-117,138)
    ////  const float MEAN_B = 123;      ( 0 - 255 ) - 123 =  (-123,132)
    ////  So image data  Range (-123,151) , Just like (-255,255)
    ////  For 256PE = 64co * 4ci * 512Depth * 9 * 16bits = 256*512*9*16bits 
    //// Layer Attributes: ( NAME       ,  W,     H,     CI,      CO,  K,PAD, STRIDE,RELU,SPLIT_1,SPLIT_2,  GPOOL, BATCHNORMAL  SUBLAYER_NUM,   SUBLAYER_SEQ,DPOSI,  DPOSO,   WPOS,  BPOS)
    addLayer(net, layer_t("conv0"       , 32,    32,      3,      32,  5,  2,      1,   0,      0,      0,      0,           0,            1,              0,    3,      3,      9,     9));
    addLayer(net, layer_t("conv1"       , 16,    16,     32,      32,  5,  2,      1,   0,      0,      0,      0,           0,            1,              0,    3,      3,      9,     9));
    addLayer(net, layer_t("conv2"       ,  8,     8,     32,      64,  5,  2,      1,   0,      0,      0,      0,           0,            1,              0,    3,      3,      9,     9));
    return net;
}
//================================================================
//=Add layer to given network
//================================================================
void addLayer(network_t *net, layer_t layer) {
// Assumes that Network has been initialized with enough memory (not checked!)
// Uses static variables -> can only be used for 1 network definition per
// program!
// If layer.isFirstSplitLayer==true, reserves double amount of
//     output memory to allow implicit output-channel concatenation.
//     Use for (expand1x1) and other "output-channel split" layers.
// If layer.isSecondSplitLayer==true, uses same input memory address as in
//     last layer, and moves output address only by layer.channels_out/2
//     to interleave with previous layer's output.
//     Use for (expand3x3) and other "output-channel split" layers.
// Keep Track of Memory Locations for Activations + Weights (-> static)
// Set variables
    static long curWeightsAddr = 0;
    static long curImgInAddr  = 0;
    static long curImgOutAddr = 0;
    float memBorder=MEMORY_ALIGNMENT;
    short poolSize=0;
    if(layer.globalPooling ==0){
    // indicate max pooling size = 2
        poolSize=2;
    }else{
    // No pooling or pooling size =1 
        poolSize=1;
    }
    int chOutAlignment  ; 
    int coPeAlignment= 0;
    if( (layer.kernel == 3) && (layer.kernel == 1)){
       coPeAlignment = PE_K3_CO; 
    }else{
       coPeAlignment = PE_K5_CO; 
    }
    chOutAlignment = CEIL_DIV(layer.channelsOut,coPeAlignment)*coPeAlignment;
    int inputImgPixels;
    if(layer.channelsIn == 3){
        inputImgPixels = layer.width * layer.height * IEMEM_1ADOTS;
    }else{
        inputImgPixels = layer.width * layer.height * layer.channelsIn; 
    }
    int widthOut=1+std::floor((float)(layer.width+2*layer.padSize-layer.kernel)/poolSize);
    int heightOut=1+std::floor((float)(layer.height+2*layer.padSize-layer.kernel)/poolSize);

    int outputImgPixels = widthOut * heightOut * layer.channelsOut;
    int numWeights=chOutAlignment * layer.channelsIn * layer.kernel * layer.kernel;
    int numBias = chOutAlignment; 
    int numScale = chOutAlignment; 
    int numRollingMean = chOutAlignment; 
    int numRollingVariance = chOutAlignment; 
    short subLayerFlag = 0;
    if(layer.sublayerNum == 1){
        // no split
        subLayerFlag = 0;
    }else { 
        if(layer.sublayerSeq == 0){
        // first split
            subLayerFlag = 1;
        }else if(layer.sublayerSeq == (layer.sublayerNum-1)){
        // end split
            subLayerFlag = 3;
        }else {
        // middle split
            subLayerFlag = 2;
        }
    }
// Set memAddrBias,scales,means,vars, and Weights , address Bytes
    if (layer.batchNorm == 1) {
    // yolo batch normal
        layer.memAddrBias = curWeightsAddr;
        curWeightsAddr += numBias*DOT_BYTES;
        layer.memAddrScale = curWeightsAddr; 
        curWeightsAddr += numScale*DOT_BYTES; 
        layer.memAddrRollingMean = curWeightsAddr;
        curWeightsAddr += numRollingMean*DOT_BYTES; 
        layer.memAddrRollingVariance = curWeightsAddr;
        curWeightsAddr += numRollingVariance*DOT_BYTES; 
        layer.memAddrWeights = curWeightsAddr; 
        curWeightsAddr += numWeights*DOT_BYTES; 
    } else { 
    // only bias, caffe  
        layer.memAddrBias = curWeightsAddr;
        layer.memAddrScale = curWeightsAddr; 
        layer.memAddrRollingMean = curWeightsAddr;
        layer.memAddrRollingVariance = curWeightsAddr;
        curWeightsAddr += numBias*DOT_BYTES;
        layer.memAddrWeights = curWeightsAddr; 
        curWeightsAddr += numWeights*DOT_BYTES; 
    }
// Set memAddrImgIn,memAddrImgOut,address Bytes
    if( subLayerFlag < 2){
    // first split
        curImgInAddr = curImgOutAddr; 
        layer.memAddrImgIn = curImgInAddr;
        curImgOutAddr +=  inputImgPixels * DOT_BYTES; 
        printf("aa = %8d\n",curImgOutAddr);
        curImgOutAddr = std::ceil(curImgOutAddr/memBorder) * memBorder;
        printf("aaNew = %8d\n",curImgOutAddr);
        layer.memAddrImgOut = curImgOutAddr;
    }else if(subLayerFlag == 2){
    // middle split
        layer.memAddrImgIn = curImgInAddr;
        //In case of large network
        //For example real total co = 1024,
        //fpga chOutAlignment = 256,
        //0) subLayerFlag = 1,
        // --> curImgOutAddr ,[0,1,2--255],[1024,1025,...].
        //1) subLayerFlag = 2,
        // --> curImgOutAddr ,[256,257---511],[1280,1281,...].
        //1) subLayerFlag = 2,
        // --> curImgOutAddr ,[512,513---767],[1536,1537,...].
        curImgOutAddr +=  chOutAlignment* DOT_BYTES; 
        layer.memAddrImgOut = curImgOutAddr;
    }else if(subLayerFlag ==3){
        //2) subLayerFlag = 3,
        // --> curImgOutAddr ,[768,769---1023],[1792,1793,...].
        layer.memAddrImgIn = curImgInAddr;
        curImgOutAddr +=  chOutAlignment* DOT_BYTES; 
        layer.memAddrImgOut = curImgOutAddr;
        // return to the first split address
        curImgOutAddr = curImgOutAddr - (layer.sublayerNum-1) * chOutAlignment * DOT_BYTES;
    }
    // Add layer to network,just like 
    net->layers[net->numLayers] = layer;
    net->numLayers++;
    assert( net->numLayers <= TOTAL_NUM_LAYERS);
    net->numImageBytes = curImgOutAddr + outputImgPixels * layer.sublayerNum * DOT_BYTES;
// Set memAddrBias,scales,means,vars, and Weights , address Bytes
    printf("layer.name                   =%8s\n", layer.name);
    printf("layer.memAddrBias            =[%08d]  0x%08xBytes\n",(long)layer.memAddrBias,(long)layer.memAddrBias);
    printf("layer.memAddrScale           =[%08d]  0x%08xBytes\n",(long)layer.memAddrScale,(long)layer.memAddrScale);
    printf("layer.memAddrRollingMean     =[%08d]  0x%08xBytes\n",(long)layer.memAddrRollingMean,(long)layer.memAddrRollingMean);
    printf("layer.memAddrRollingVariance =[%08d]  0x%08xBytes\n",(long)layer.memAddrRollingVariance,(long)layer.memAddrRollingVariance);
    printf("layer.memAddrWeights          =[%08d]  0x%08xBytes\n",(long)layer.memAddrWeights,(long)layer.memAddrWeights);
    printf("End Weights Addr             =[%08d]  0x%08xBytes\n",(long)curWeightsAddr,(long)curWeightsAddr);
    printf("|--------|\n");
    printf("layer.memAddrImgIn           =[%08d]  0x%08xBytes\n", (long)layer.memAddrImgIn, (long)layer.memAddrImgIn);
    printf("layer.memAddrImgOut          =[%08d]  0x%08xBytes\n", (long)layer.memAddrImgOut,(long)layer.memAddrImgOut);
    printf("net->numImageBytes           =[%08d]  0x%08xBytes\n", (long)net->numImageBytes,(long)net->numImageBytes);
    printf("inputImgPixels               =[%08d]  0x%08xDots\n",  (long)inputImgPixels, (long)inputImgPixels);
    printf("outputImgPixels              =[%08d]  0x%08xDots\n",  (long)outputImgPixels, (long)outputImgPixels);
    printf("End output Addr              =[%08d]  0x%08xBytes\n", (long)(net->numImageBytes),(long)(net->numImageBytes));
    printf("curImgOutAddr                =%16x\n", (int)curImgOutAddr);
    printf("curImgInAddr                 =%16x\n", (int)curImgInAddr);
    printf("||**********************************************||\n");
}
//================================================================
//=Load weights data from file
//================================================================
void loadWeightsFromFile(network_t *net,const char *fileName){
    printf("===============================================================\n");  
    printf("===============================================================\n");  
    printf("WeightsFile Formate...                                         \n");  
    printf("(1)Tiny Yolo Formate...                                        \n");  
    printf("Size (Float) Class Org        Class New                        \n");
    printf("4        VersionInfo    ||                                     \n");  
    printf("Chout    BiasChout      ||    BiasChout                        \n");  
    printf("Chout    ScalesChout    ||--> ScalesChout                      \n");  
    printf("Chout    MeansChout     ||    MeansChout                       \n");  
    printf("Chout    VarianceChout  ||    VarianceChout                    \n");  
    printf("ChoutxChinxK Weights    ||    Weights                          \n");  
    printf("***************************************************************\n");  
    printf("(2)Caffe Weights Formate...                                    \n");  
    printf("(2.1)BiasChout          ||          BiasChout                  \n");  
    printf("=====================   ||          ========================   \n");                          
    printf("bias co0,co1,...con     ||          bias co0,co1,...con        \n");                         
    printf("=====================   ||          ========================   \n");                               
    printf("(2.2)Chout * Chin * k WeightsNormal                            \n"); 
    printf("CaffeWeightsFormat      ||          FPGA CNN Format            \n"); 
    printf("=====================   ||          ========================   \n");                               
    printf("co means weights output channels ------------------------------\n");
    printf("ci means weights input channels--------------------------------\n");
    printf("k  means weights kernel ---------------------------------------\n");
    printf("co0,ci0,k0,k1,k2,..kn   ||          ci0,k0,co0,co1,co2,..con   \n");                         
    printf("co0,ci1,k0,k1,k2,..kn   ||          ci0,k1,co0,co1,co2,..con   \n");                         
    printf("co0,ci2,k0,k1,k2,..kn   ||          ci0,k2,co0,co1,co2,..con   \n");                         
    printf(".....................   ||          ........................   \n");                         
    printf("co0,cin,k0,k1,k2,..kn   ||          ci0,kn,co0,co1,co2,..con   \n");                         
    printf("---------------------   ||          ------------------------   \n");                         
    printf("co1,ci0,k0,k1,k2,..kn   ||          ci1,k0,co0,co1,co2,..con   \n");                            
    printf("co1,ci1,k0,k1,k2,..kn   ||     \\\\   ci1,k1,co0,co1,co2,..con \n");                               
    printf("co1,ci2,k0,k1,k2,..kn   || =====\\\\  ci1,k2,co0,co1,co2,..con \n");                                     
    printf(".....................   || =====//  ........................   \n");                               
    printf("co1,cin,k0,k1,k2,..kn   ||     //   ci1,kn,co0,co1,co2,..con   \n");
    printf("---------------------   ||          ------------------------   \n");                         
    printf("---------------------   ||          ------------------------   \n");                         
    printf("---------------------   ||          ------------------------   \n");                         
    printf("---------------------   ||          ------------------------   \n");                         
    printf("con,ci0,k0,k1,k2,..kn   ||          cin,k0,co0,co1,co2,..con   \n");                         
    printf("con,ci1,k0,k1,k2,..kn   ||          cin,k1,co0,co1,co2,..con   \n");                         
    printf("con,ci2,k0,k1,k2,..kn   ||          cin,k2,co0,co1,co2,..con   \n");                         
    printf(".....................   ||          .....................      \n");                         
    printf("con,cin,k0,k1,k2,..kn   ||          cin,kn,co0,co1,co2,..con   \n");                         
    FILE *fileHandle = fopen(fileName,"rb");
    if(!fileHandle){
        printf("ERROR: File %s could not be opened!\n",fileName);
        exit(-1);
    }
    cpu_data_t biasArray[MAX_CHANNELS]={0};
    for(int i = 0; i < net->numLayers; i++){
        layer_t *layer = &net->layers[i];
        int chOut = layer->channelsOut  ;
        int chIn  = layer->channelsIn   ;
        int kernelSquare = layer->kernel * layer->kernel;
        cpu_data_t weightsTimes = (1 << layer->weightsPos);
        cpu_data_t biasTimes = (1 << layer->biasPos);
        int numBias = chOut;
        int numWeights = chOut * chIn * kernelSquare;
        cpu_data_t *biasCpuAddr = (cpu_data_t *)malloc( sizeof(cpu_data_t) * numBias );
        cpu_data_t *weightsCpuAddr = (cpu_data_t *)malloc( sizeof(cpu_data_t) * numWeights );
        // load bias
        layer_weights_t *biasLayerAddr = net->weightsBaseAddr + layer->memAddrBias;
        transformCommonBiasData(fileHandle,numBias,biasTimes,biasCpuAddr,biasLayerAddr);
        // load weightss 
        layer_weights_t *weightsLayerAddr = net->weightsBaseAddr + layer->memAddrWeights;
        transformWeightsData(fileHandle,numWeights,weightsTimes,weightsCpuAddr,weightsLayerAddr,layer);
        free(biasCpuAddr);
        free(weightsCpuAddr);
    }
}
void transformWeightsData(
        FILE *fileHandle,
        int numWeights,
        cpu_data_t weightsTimes,
        cpu_data_t *weightsCpuAddr,
        layer_weights_t *weightsLayerAddr,
        layer_t *layer){
	//Convert the file data to weightsCpuAddr
    cpu_data_t *weightsCpuAddrCnt = weightsCpuAddr;
    for(int i = 0;i < numWeights ; i++){
    	fscanf(fileHandle, "%f",weightsCpuAddrCnt++);
    }
    //Transform
    int chOut = layer->channelsOut;
    int chIn  = layer->channelsIn;
    int kernelSquare = layer->kernel * layer->kernel;
    int chOutAlignment  ; 
    int coPeAlignment= 0;
    if( (layer->kernel == 3) && (layer->kernel == 1)){
       coPeAlignment = PE_K3_CO; 
    }else{
       coPeAlignment = PE_K5_CO; 
    }
    chOutAlignment = CEIL_DIV(layer->channelsOut,coPeAlignment)*coPeAlignment;
    int weightsLayerAddrOffset = 0;
    for(int ci = 0; ci < chIn; ci++){
        for(int kernel = 0 ; kernel < kernelSquare ; kernel++){
            // only support co == 8 ,kernel 5x5, not support co =64, kernel 3x3
            weightsLayerAddrOffset = ( kernel * chOutAlignment + ci * kernelSquare * chOutAlignment)*DOT_BYTES;
            uatom_t *apfixWp = (uatom_t *)(weightsLayerAddr + weightsLayerAddrOffset);
            for(int co = 0; co < chOutAlignment; co++){
            	cpu_data_t weightsFloat ;
                atom_t dataTmp = round((*(weightsCpuAddr + co * kernelSquare * chIn + kernel + ci * kernelSquare)) * weightsTimes );
                *apfixWp++ = *(uatom_t *) &dataTmp;
            }
        }
    }
}
void transformCommonBiasData(
    FILE *fileHandle,
    int numBias,
    cpu_data_t biasTimes,
    cpu_data_t *biasCpuAddr,
    layer_weights_t *biasLayerAddr){
	//Convert the file data to weightsCpuAddr
    cpu_data_t *weightsCpuAddrCnt = biasCpuAddr;
    for(int i = 0;i < numBias ; i++){
    	fscanf(fileHandle, "%f",weightsCpuAddrCnt++);
    }
    //Transform
    uatom_t *apfixWp = (uatom_t *)(biasLayerAddr);
    for(int co = 0; co < numBias; co++){
        atom_t dataTmp = round( (*(biasCpuAddr+co)) * biasTimes);
        *apfixWp++ = *(uatom_t *)&dataTmp;
    }
}       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

