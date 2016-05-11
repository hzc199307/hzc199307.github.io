/******************************************************************** 
##### File Name: squareSum2.cu 
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
    generateData(data, DATA_SIZE);
    // malloc space for datas in GPU
    int *gpuData, *sum;
    clock_t *time;
    cudaMalloc(( void **) &gpuData, sizeof ( int ) * DATA_SIZE);
    cudaMalloc(( void **) &sum, sizeof ( int ));
    cudaMalloc(( void **) &time, sizeof (clock_t));
    cudaMemcpy(gpuData, data, sizeof ( int ) * DATA_SIZE, cudaMemcpyHostToDevice);
    // calculate the squares's sum
    //CUDA调用在GPU中函数名称<<<block num, thread num, shared memory size>>>(param,...) ;
    squaresSum<<< 1 , 1 , 0 >>>(gpuData, sum, time);
    // copy the result from GPU to HOST
    int result;
    clock_t time_used;
    cudaMemcpy(&result, sum, sizeof ( int ), cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof (clock_t), cudaMemcpyDeviceToHost);
    // free GPU spaces
    cudaFree(gpuData);
    cudaFree(sum); cudaFree(time);
    // print result
    printf ( "(GPU) sum:%d time:%ld\n" , result, time_used);
    // CPU calculate
    result = 0 ;
    clock_t start = clock();
    for ( int i = 0 ; i < DATA_SIZE; ++i) {
        result += data[i] * data[i];
    }
    time_used = clock() - start;
    printf ( "(CPU) sum:%d time:%ld\n" , result, time_used);
    return 0 ;
}
//__global__ means that this function run in GPU, there isn't any return value.
__global__ static void squaresSum( int *data, int *sum, clock_t *time) {
    int sum_t = 0 ;
    clock_t start = clock();
    for ( int i = 0 ; i < DATA_SIZE; ++i) {
        sum_t += data[i] * data[i];
    }
    *sum = sum_t;
    *time = clock() - start;
}
// ======== used to generate rand datas ========
void generateData( int *data, int size) {
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
    cudaSetDevice(0);
    printf ( "Device No%d is selected.\n",0 );

    return true;
}
