/******************************************************************** 
##### File Name: squareSum6.cu 
##### File Func: calculate the sum of inputs's square
##### Author: HeZhichao 
##### E-mail: hzc199307@gmail.com 
##### Create Time: 2016-5-11 
*********************************************************************/
# include<stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
// ======== define area ========
# define DATA_SIZE 1048576 // 1M
//8*128=1024 threads
# define BLOCK_NUM 8 // block num
# define THREAD_NUM 128 // thread num per block

// ======== global area ========
int data[DATA_SIZE];
void printDeviceProp( const cudaDeviceProp &prop);
bool InitCUDA(); 
void generateData( int *data, int size);
__global__ static void squaresSum( int *data, int *sum, clock_t *time);
int main( int argc, char const *argv[]) {
    // init CUDA device
    if (!InitCUDA()) { return 0 ; }
    printf ( "CUDA initialized.\n" );
    // generate rand datas 
    printf("test !\n");
    generateData(data, DATA_SIZE);
    // malloc space for datas in GPU
    int *gpuData;
    int *sum;
    clock_t *time;
    printf("cudaMalloc start !\n");
    cudaMalloc(( void **) &gpuData, sizeof ( int ) * DATA_SIZE);
    printf("cudaMalloc gpuData is ok !\n");
    cudaMalloc(( void **) &sum, sizeof ( int )*BLOCK_NUM);
    printf("cudaMalloc sum is ok !\n");
    cudaMalloc(( void **) &time, sizeof (clock_t));
    cudaMemcpy(gpuData, data, sizeof ( int ) * DATA_SIZE, cudaMemcpyHostToDevice);
    printf("cudaMemcpy data to gpuData is ok !\n");
    // calculate the squares's sum
    //CUDA调用在GPU中函数名称<<<block num, thread num, shared memory size>>>(param,...) ;
    squaresSum<<< BLOCK_NUM , THREAD_NUM , THREAD_NUM*sizeof(int) >>>(gpuData, sum, time);
    // copy the result from GPU to HOST
    int result[BLOCK_NUM];
    clock_t time_used;
    cudaMemcpy(result, sum, sizeof ( int )*BLOCK_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof (clock_t), cudaMemcpyDeviceToHost);
    // free GPU spaces
    cudaFree(gpuData);
    cudaFree(sum); cudaFree(time);
    // print result
    int tmp_result = 0;
    for(int i=0;i<BLOCK_NUM;++i){
        tmp_result += result[i];
    }
    printf ( "(GPU) sum:%d time:%ld\n" , tmp_result, time_used);
    // CPU calculate
    tmp_result = 0 ;
    clock_t start = clock();
    for ( int i = 0 ; i < DATA_SIZE; ++i) {
        tmp_result += data[i] * data[i];
    }
    time_used = clock() - start;
    printf ( "(CPU) sum:%d time:%ld\n" , tmp_result, time_used);/**/
    return 0 ;
}
//__global__ means that this function run in GPU, there isn't any return value.
__global__ static void squaresSum( int *data, int *sum, clock_t *time) {
    // define of shared memory
    __shared__ int shared[THREAD_NUM];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    shared[tid] = 0 ;
    
    clock_t start = clock();

    for ( int i = bid * THREAD_NUM + tid ; i < DATA_SIZE; i+=THREAD_NUM*BLOCK_NUM) {
        shared[tid] += data[i] * data[i];
    }
    
    //同步操作，必须等到之前的线程都运行结束，才能继续后面的程序
    __syncthreads();
    //同步完成之后，将部分和加到share[0]上面
    if(tid==0){ //这里保证全部都在一个线程内完成
        for(int i=1;i<THREAD_NUM;i++){
            shared[0]+=shared[i];
        }
        sum[bid]=shared[0];
    }

    *time = clock() - start;
}
// ======== used to generate rand datas ========
void generateData( int *data, int size) {
    printf("generateData !");
    for ( int i = 0 ; i < size; ++i) {
        data[i] = rand() % 10 ;
    }
}
void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %lu.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %lu.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %lu.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %lu.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %lu.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

bool InitCUDA()
{
    //used to count the device numbers
    int count;

    // get the cuda device count
    cudaGetDeviceCount(&count);
    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    // find the device >= 1.X
    bool noDeviceSupport = true;
    int i;
    for (i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
		noDeviceSupport = false;
		printf("****** Device No%d*********************************\n",i);
		printDeviceProp(prop);
		printf("\n");
            }
        }
    }

    // if can't find the device
    if (noDeviceSupport == true) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    // set cuda device
    cudaSetDevice(4);
    printf ( "Device No%d is selected.\n",4 );

    return true;
}
