---
title: CUDA入门
date: 2016-05-11 00:00:00
categories: 
	- Tech
	- GPU
tags: [GPU,CUDA]
copyright: true
---

CUDA安装在网上的教程很多，这里就不再详述。此外，笔者也是今天才开始写cuda的代码，所以内容上只是简单总结。

<!-- more -->

# 初识CUDA

CUDA 目前有两种不同的 API：Runtime API 和 Driver API，两种 API 各有其适用的范围。由于 runtime API 较容易使用，一开始我们会以 runetime API 为主。所以代码开头加上#include <cuda_runtime.h>。

以为CUDA语言基于C，里面涉及到很多指针的操作，C不熟悉的话可以先复习一下指针的使用。

## 部分函数介绍

1.cudaError_t cudaGetDeviceCount( int* count )
通过count返回可用于计算的设备数量。

2.cudaError_t cudaGetDeviceProperties( struct cudaDeviceProp* prop,int dev )
通过prop返回第dev台设备的属性，dev编号从0开始。

3.cudaError_t cudaSetDevice(int dev)
设置第dev台为执行设备。

## cudaDeviceProp结构

	struct cudaDeviceProp {

		char name [256];			//用于标识设备的ASCII字符串;
		size_t totalGlobalMem;		//设备上可用的全局存储器的总量,以字节为单位;
		size_t sharedMemPerBlock;	/*线程块可以使用的共享存储器的最大值,以字节为单位;
									  多处理器上的所有线程块可以同时共享这些存储器;*/
		int regsPerBlock;			/*线程块可以使用的32位寄存器的最大值;
									多处理器上的所有线程块可以同时共享这些寄存器;*/
		int warpSize;				//按线程计算的warp块大小;
		size_t memPitch;			/*允许通过cudaMallocPitch()为包含存储器区域的
									存储器复制函数分配的最大间距(pitch),以字节为单位;*/
		int maxThreadsPerBlock;		//每个块中的最大线程数
		int maxThreadsDim [3];		//块各个维度的最大值:
		int maxGridSize [3];		//网格各个维度的最大值;
		size_t totalConstMem;		//设备上可用的不变存储器总量,以字节为单位;
		int major;					//定义设备计算能力的主要修订号和次要修订号;
		int minor;					//
		int clockRate;				//以千赫为单位的时钟频率;
		size_t textureAlignment;	/*对齐要求;与textureAlignment字节对齐的
									纹理基址无需对纹理取样应用偏移;*/
		int deviceOverlap;			/*如果设备可在主机和设备之间并发复制存储器,
									同时又能执行内核,则此值为 1;否则此值为 0;*/
		int multiProcessorCount;	//设备上多处理器的数量。

	}

## 编译

nvcc 是 CUDA 的编译工具，它可以将 .cu 文件解析出在 GPU 和 host 上执行的部分.也就是说，它会帮忙把 GPU 上执行和主机上执行的代码区分开来，不需要我们手动去做了。在 GPU 执行的部分会通过 NVIDIA 提供的 编译器编译成中介码，主机执行的部分则调用 gcc 编译。
{% codeblock lang:c %}
nvcc -o first_cuda first_cuda.cu
{% endcodeblock %}

# 使用GPU实现数组平方和

## 用GPU的简单实现
{% include_code1 lang:c squareSum2.cu %}

## 改进1：多线程
{% include_code lang:c squareSum3.cu %}

## 改进2：使内存的整体读取连续
{% include_code lang:c squareSum4.cu %}

## 改进3：多block多thread
{% include_code lang:c squareSum5.cu %}

## 改进4：使用block的共享内存
以在block上求和该block的所有thread，然后再CPU中求和（小步提升）。
{% include_code lang:c squareSum6.cu %}

## 改进5：加法树
在block内部求和时可以采用加法树的方法，实现并行化（小步提升）。
{% include_code lang:c squareSum7.cu %}
