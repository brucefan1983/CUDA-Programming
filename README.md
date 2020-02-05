# 《`CUDA` 编程：基础与实践》源代码

## 1. 告读者：
* 代码还在开发中。由琪同学为本书写了 Python 版本的代码（用 pyCUDA）:
https://github.com/YouQixiaowu/CUDA-Programming-with-Python

## 2. 关于本书：
  * 将于 2020 年在清华大学出版社出版，语言为中文。
  * 覆盖开普勒到图灵（计算能力从 3.5 到 7.5）的所有 GPU 架构。
  * 尽量同时照顾 Windows 和 Linux 用户。
  * 假设读者有如下基础：
    * 熟悉 `C++` (对全书来说)；
    * 熟悉本科水平的物理和数学（对某些章节来说）。
    
## 3. 我的测试系统
* Linux: 主机编译器用的 `g++`。
* Windows: 仅使用命令行解释器 `CMD`，主机编译器用 Visual Studio 中的 `cl`。在用 `nvcc` 编译 CUDA 程序时，可能需要添加 `-Xcompiler "/wd 4819"` 选线消除和 unicode 有关的警告。
* 全书代码可在 `CUDA` 9-10.2 （包含）之间的版本运行。


## 4. 目录和源代码条目

### 第 1 章：GPU 硬件和 CUDA 工具

本章无源代码。


### 第 2 章：`CUDA` 中的线程组织

| 文件       | 知识点 |
|:------------|:---------------|
| `hello.cpp` | 用 `C++` 写一个 Hello World 程序 |
| `hello1.cu` | 一个正确的 `C++` 程序也是一个正确的 `CUDA` 程序 | 
| `hello2.cu` | 写一个打印字符串的 `CUDA` 核函数并调用 | 
| `hello3.cu` | 使用含有多个线程的线程块 |
| `hello4.cu` | 使用多个网格 |
| `hello5.cu` | 使用两维线程块 |


### 第 3 章：`CUDA` 程序的基本框架

| 文件        | 知识点 |
|:------------|:---------------|
| `add.cpp`      | 数组相加的 `C++` 版本 |
| `add1.cu`      | 数组相加的 `CUDA` 版本 |
| `add2wrong.cu` | 如果数据传输方向搞错了会怎样？ |
| `add3if.cu`    | 什么时候必须在核函数使用 if 语句？ |
| `add4device.cu`| 定义与使用 `__device__` 函数 |


### 第 4 章：`CUDA` 程序的错误检测

| 文件       | 知识点 |
|:------------|:---------------|
| `check1api.cu`    | 检测 `CUDA` 运行时 API 函数的调用 |
| `check2kernel.cu` | 检测 `CUDA` 核函数的调用 | 
| `memcheck.cu`     | 用 `cuda-memcheck` 检测内存方面的错误 |
| `error.cuh`       | 本书常用的用于检测错误的宏函数 |


### 第 5 章：获得 GPU 加速的前提

| 文件       | 知识点 |
|:------------|:---------------|
| `add1cpu.cu`    | 为 `C++` 版的数组相加函数计时 |
| `add2gpu.cu`    | 为数组相加核函数计时 |
| `add3memcpy.cu` | 如果把数据传输的时间也包含进来，还有加速吗？|
| `arithmetic1cpu.cu`       | 提高算术强度的 `C++` 函数 | 
| `arithmetic2gpu.cu`       | 提高算术强度的核函数；GPU/CPU 加速比是不是很高？ |


### 第 6 章： `CUDA` 中的内存组织

| 文件        | 知识点 | 
|:------------|:---------------|
| `static.cu`    | 如何使用静态全局内存 |
| `query.cu`     | 如何在 CUDA 程序中查询所用 GPU 的相关技术指标 |


### 第 7 章：全局内存的合理使用

| 文件        | 知识点 | 
|:------------|:---------------|
| `matrix.cu` | 合并与非合并读、写对程序性能的影响 |

### 第 8 章：共享内存的合理使用

| 文件        | 知识点 |
|:------------|:---------------|
| `reduce1cpu.cu`     | `C++` 版本的规约函数 |
| `reduce2gpu.cu`     | 仅使用全局内存和同时使用全局内存和共享内存的规约核函数|
| `bank.cu`           | 使用共享内存实现矩阵转置并避免共享内存的 bank 冲突 |


### 第 9 章：原子函数的合理使用

| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `reduce.cu`        | using `atomicAdd` for reduction (good or bad?) |
| `neighbor_cpu.cu`  | neighbor list construction using CPU |
| `neighbor_gpu.cu`  | neighbor list construction using GPU, with and without using `atomicAdd` |


### 第 10 章: 线程束内部函数
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `reduce7syncwarp.cu` | using the `__syncwarp()` function instead of the `__syncthreads()` function within warps |
| `reduce8shfl.cu`     | using the `__shfl_down_sync()` or the `__shfl_xor_sync()` function for warp reduction |
| `reduce9cp.cu`       | using the cooperative groups |


### 第 11 章： `CUDA` 流
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `host_kernel.cu`     | overlapping host and device computations |
| `kernel_kernel.cu`   | overlaping multiple kernels |
| `kernel_transfer.cu` | overlaping kernel execution and memory transfer |


### Chapter 12: Using `CUDA` libraries
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `thrust_scan_vector.cu`  | using the device vector in `thrust` |
| `thrust_scan_pointer.cu` | using the device pointer in `thrust` |
| `cublas_gemm.cu`         | matrix multiplication in `cuBLAS` |
| `cusolver.cu`            | matrix eigenvalues in `cuSolver` |
| `curand_host1.cu`        | uniform random numbers in `cuRAND` |
| `curand_host2.cu`        | Gaussian random numbers in `cuRAND` |


### Chapter 13: Unified memory programming 
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `add_unified.cu` | using unified memory |


### Chapter 14: Summary for developing and optimizing `CUDA` programs 
There is no source code for this chapter.


### Chapter 15: Introduction to molecular dynamics simulation
There is no source code for this chapter.


### Chapter 16: A `C++` program for molecular dynamics simulation
How to compile and run?
  * type `make` to compile
  * type `./ljmd 8 10000` to run
  * type `plot_results` in Matlab command window to check the results


### Chapter 17: MD code: only accelerating the force-evaluation part
How to compile and run?
  * type `make` to compile
  * type `./ljmd 40 10000` to run
  * type `plot_results` in Matlab command window to check the results


### Chapter 18: MD code: accelerating the whole code
How to compile and run?
  * type `make` to compile
  * type `./ljmd 40 10000` to run
  * type `plot_results` in Matlab command window to check the results


### Chapter 19: MD code: various optimizations
How to compile and run?
  * type `make` or `make -f makefile.ldg` or `make -f makefile.fast_math` to compile
  * type `./ljmd 40 10000` to run
  * type `plot_results` in Matlab command window to check the results


### Chapter 20: MD code: using unified memory
How to compile and run?
  * type `make` or `make -f makefile.pascal` to compile
  * type `./ljmd 40 10000` to run
  * type `plot_results` in Matlab command window to check the results
  
  
## 5. 我的部分测试结果

### 4.1. 矢量相加 (第 5 章)

* 数组元素个数 = 1.0e8。
* CPU (我的笔记本) 函数的执行时间是 60 ms （单精度）核 120 ms （双精度）。
* GPU 执行时间见下表：

|  V100 (S) | V100 (D) | 2080ti (S) | 2080ti (D) | P100 (S) | P100 (D) | laptop-2070 (S) | laptop-2070 (D) | K40 (S) | K40 (D) |
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
| 1.5 ms | 3.0 ms |  2.1 ms |  4.3 ms | 2.2 ms |  4.3 ms | 3.3 ms | 6.8 ms | 6.5 ms | 13 ms |

* 如果包含 cudaMemcpy 所花时间，GeForce RTX 2070-laptop 用时 180 ms （单精度）和 360 ms （双精度），是 CPU 版本的三倍慢！

### 4.2. 一个高算术强度的函数（第 5 章）
* CPU 函数（数组长度为 10^4）用时 320 ms （单精度）和 450 ms （双精度）。
* GPU 函数（数组长度为 10^6）用时情况如下表：

|  V100 (S) | V100 (D) | 2080ti (S) | 2080ti (D) | laptop-2070 (S) | laptop-2070 (D) |
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
| 11 ms | 28 ms |  15 ms |  450 ms | 28 ms |  1000 ms |

* 用 GeForce RTX 2070-laptop 时核函数执行时间与数组元素个数 N 的关系见下表（单精度）：

| N    | 时间 |
|:-------|:-------|
| 1000    | 0.91 ms | 
| 10000   | 0.99 ms | 
| 100000  | 3.8 ms | 
| 1000000 | 28 ms |
| 10000000   | 250 ms | 
| 100000000  | 2500 ms |

### 4.3. 矩阵复制和转置 (第 7-8 章)
* 矩阵维度为 10000 乘 10000。
* 核函数执行时间见下表：

| 计算     | V100 (S) | V100 (D) | 2080ti (S) | 2080ti (D) | K40 (S) |
|:---------------------------------|:-------|:-------|:-------|:-------|:-------|
| 矩阵复制                             | 1.1 ms | 2.0 ms | 1.6 ms | 2.9 ms |  | 
| 读取为合并、写入为非合并的矩阵转置     | 4.5 ms | 6.2 ms | 5.3 ms | 5.4 ms | 12 ms | 
| 写入为合并、读取为非合并的矩阵转置     | 1.6 ms | 2.2 ms | 2.8 ms | 3.7 ms | 23 ms | 
| 在上一个版本的基础上使用 `__ldg` 函数 | 1.6 ms | 2.2 ms | 2.8 ms | 3.7 ms | 8 ms |
| 利用共享内存转置，但有 bank 冲突      | 1.8 ms | 2.6 ms | 3.5 ms | 4.3 ms |  | 
| 利用共享内存转置，且无 bank 冲突      | 1.4 ms | 2.5 ms | 2.3 ms | 4.2 ms |  |


### 4.4. Reduction (chapters 8-10 and 14)

* Array length = 1.0e8 and each element has a value of 1.23.
* The correct summation should be 123000000.
* Using single precision with both CPU and GPU (Tesla K40).

| computation & machine                         | time    |   result  |
|:----------------------------------------------|:--------|:----------|
| CPU with naive summation                      | 85 ms   | 33554432  | 
| global memory only                            | 16.3 ms | 123633392 | 
| static shared memory                          | 10.8 ms | 123633392 | 
| dynamic shared memory                         | 10.8 ms | 123633392 |  
| atomicAdd                                     | 9.8 ms  | 123633392 | 
| atomicAdd and syncwarp                        | 8.1 ms  | 123633392 | 
| atomicAdd and shfl                            | 6.3 ms  | 123633392 | 
| atomicAdd and CP                              | 6.3 ms  | 123633392 | 
| two kernels and less blocks                   | 2.8 ms  | 122999920 | 
| two kernels and less blocks and no cudaMalloc | 2.6 ms  | 122999920 |


### 4.5. Neighbor list construction (chapter 9)

* Number of atoms = 22464.
* CPU function takes 230 ms for both single and double precisions.
* GPU timing results are list in the following table:

| computation     | V100 (S) | V100 (D) | K40 (S) | K40 (D) | 
|:----------------|:---------|:---------|:-----------|:-----------|
| neighbor without atomicAdd | 1.9 ms | 2.6  ms | 10.1 ms | 10.9 ms |
| neighbor with atomicAdd    | 1.8 ms | 2.6  ms | 10.5 ms | 14.5 ms |






