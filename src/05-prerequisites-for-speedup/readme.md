# Chapter 5 Prerequisites for obtaining high performance in CUDA programs



In the previous chapters, we have only discussed the correctness of a CUDA program. Starting from this chapter, we will focus on the performance of CUDA programs. 

## 5.1 Using CUDA events to time a block of code

There are many methods of timing for a block of code in a CUDA program, but here we only introduce a method based on CUDA events:

```c++
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));
CHECK(cudaEventRecord(start));
cudaEventQuery(start); // cannot use the macro function CHECK here

// The code block to be timed

CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));
float elapsed_time;
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
printf("Time = %g ms.\n", elapsed_time);

CHECK(cudaEventDestroy(start));
CHECK(cudaEventDestroy(stop));
```

This timing method can be understood as follows:

* First, we define two CUDA events, `start` and `stop`, which are of type `cudaEvent_t`, and then initialize them using the `cudaEventCreate` function.
* Next, we pass `start` into the function `cudaEventRecord` to record a time stamp representing the start of the code block to be timed. The next call to `cudaEventRecord` is only necessary for GPUs in the WDDM mode.
* Then, after the code block, we pass `stop` into the function `cudaEventRecord` to record a time stamp representing the end of the code block. The next call to `cudaEventSynchronize` forces the host to wait for the completion of the previous statement.
* Then, we use the function `cudaEventElapsedTime` to calculate the time interval `elapsed_time` between `stop` and `start`, in units of ms (micro second, or 1/1000 second).
* The last two lines are used to clean up resources. 

We first use this method time time the `add1cpu.cu` program of this chapter, which is adapted from the `add.cpp` program of chapter 3. We can use the following command to compile a version using single-precision floating point numbers in the code:

```shell
$ nvcc -O3 -arch=sm_75 add1cpu.cu
```

If we want to get a version using double-precision floating point numbers, we can use the the following command:

```shell
$ nvcc -O3 -arch=sm_75 -DUSE_DP add1cpu.cu
```

We can similarly build the single-precision and double-precision versions of the corresponding CUDA program `add2gpu.cu` of this chapter. Using single-precision, the host function `add` in `add1cpu.cu` takes about 60 ms and the CUDA kernel `add` in `add2gpu.cu` takes about 3.3 ms. Using double precision, they take about 120 ms and 6.8 ms, respectively. The author also tested the CUDA program `add2gpu.cu` using other GPUs, including K40, P100, V100, and RTX 2080ti. The relevant timing results are presented in the following table:

| V100 (S) | V100 (D) | 2080ti (S) | 2080ti (D) | P100 (S) | P100 (D) | 2070 (S) | 2070 (D) | K40 (S) | K40 (D) |
| :------- | :------- | :--------- | :--------- | :------- | :------- | :------- | :------- | :------ | :------ |
| 1.5 ms   | 3.0 ms   | 2.1 ms     | 4.3 ms     | 2.2 ms   | 4.3 ms   | 3.3 ms   | 6.8 ms   | 6.5 ms  | 13 ms   |

From the testing results, we see that for all the machines (CPU or GPUs), the double-precision version is about 2X as slow as the corresponding single-precision version. This is because the array addition problem is dominated by memory accessing, rather than floating point computations.

We can also calculate the effective memory bandwidth of a task, which is defined as the accessed number of bytes divided by time used. Taking RTX 2070 and single-precision as an example, the effective band width is 3 * 1.0e8 * 4 B / 3.3 ms ~ 360 GB/s, which is slightly smaller than the theoretical band width of this GPU (448 GB/s). When the effective band width of a task is close to the theoretical band width, it indicates that the task is memory accessing bounded, not floating point computation bounded. The performance (inverse of time) of the `add` kernel is roughly proportional to the theoretical band width of the GPU.

## 5.2 Factors affecting GPU acceleration

Using RTX 2070, the speedup factor of the CUDA kernel over the corresponding host function is 60/3.3, which is about 17. This is not very low, but is also not very high. In this section, we discuss when a high speedup factor can be obtained.

### 5.2.1 Ratio of data transfer

In the program `add2gpu.cu`, we have only timed the CUDA kernel. Here, we also include the data transfer before and after the kernel into the code block to be timed, as in the program `add3memcpy.cu`. Using RTX 2070, this part takes 180 ms and 360 ms, respectively, for the single-precision and double-precision versions. We see that if we include the time spent on data transfer, the CUDA program is even 3 times as slow as the C++ program. The reason is that the bandwidth of GPU memory accessing is more than one order of magnitude higher than the bandwidth of data transfer between CPU and GPU (through PCIe). 

In a realistic CUDA program, the whole task should never be adding up to arrays only. That would be a very stupid CUDA application. There must be more calculations in the device with more kernel invocations. One of the most important principles for CUDA programming is to minimize the amount of data transfer between host and device. For most of the examples in this book, however, the time spending on the kernel might be a small fraction of the time for whole CUDA program, and the reader should understand that the major purpose of these examples is to demonstrate how to optimize the kernel performance. 

### 5.2.2 Arithmetic intensity

The speedup factor in the array addition problem is not very high (only considering the kernel and the host function, as we remarked above), which is mainly due to the low **arithmetic intensity** in this problem. The **arithmetic intensity** of a problem refers to the ratio between the amount of arithmetic operations and the amount of the memory operations that are used to support the arithmetic operations. The arithmetic intensity of the array addition problem is quite low, because there is a single floating point addition with 3 global memory accessing events (two reads and one write). The function `arithmetic` in the programs `arithmetic1cpu.cu` and `arithmetic2gpu.cu` has much larger arithmetic intensity:

```c++
const real x0 = 100.0;
void __global__ arithmetic(real *d_x, const real x0, const int N)
{ 
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        real x_tmp = d_x[n];
        while (sqrt(x_tmp) < x0)
        {
            ++x_tmp;
        }
        d_x[n] = x_tmp;
    }
}
```

When the array length is 10 000, the host function takes 320 ms and 450 ms, using single and double precision floating point numbers, respectively. When the array length is 1000 000, the kernel with GeForce RTX 2070 takes 28 ms and 1000 ms, using single and double precision floating point numbers, respectively. Therefore, the speedup factor is about 1100 and 45 using single and double precision floating point numbers, respectively. We also note that in this case, using single-precision floating point numbers is much faster, which is a strong indicator of the high arithmetic intensity of the kernel. If we change to use Tesla V100 to test, the times used by the kernel are 11 ms (single precision) and 28 ms (double precision), showing much less difference between single and double precision floating point numbers.

### 5.2.3 Degree of parallelism 

The **degree of parallelism** for a CUDA kernel is essentially the total number of threads assigned for the kernel. Each GPU consists of multiple streaming multiprocessors (SM) and each SM has a number of CUDA cores. Each SM can support about 1024 parallel threads and a typical GPU can thus support at least tens of thousands of parallel threads. If the number of threads assigned for a CUDA kernel is much smaller than this number, it would be hard to get high performance.

![Effects of the degree of parallelism on the performance of the kernel](speed.png)



为了验证这个论断，我们将~\verb"arithmetic2gpu.cu"~程序中的数组元素个数~\verb"N"~从~$10^3$~以10倍的间隔增加到~$10^8$，分别测试核函数的执行时间，结果展示在图~\ref{figure:data_size}（a）中。因为~CPU~中的计算时间基本上与数据量成正比，所以我们可以根据之前的结果计算~\verb"N"~取不同值时~GPU~程序相对于~CPU~程序的加速比，结果显示在图~\ref{figure:data_size}（b）中。

由图~\ref{figure:data_size}（a） 可知，在数组元素个数~\verb"N"~很大时，核函数的计算时间正比于~\verb"N"；在~\verb"N"~很小时，核函数的计算时间不依赖于~\verb"N"~ 的值，保持为常数。这两个极限情况都是容易理解的。当~\verb"N" 足够大时，GPU~是满负荷工作的，增加一倍的工作量就会增加一倍的计算时间。反之，当~\verb"N"~不够大时，GPU~中是有空闲的计算资源的，增加~\verb"N"~的值并不会增加计算时间。若要让~GPU~满负荷工作，则核函数中定义的线程总数要不少于某个值，该值在一般情况下和~GPU~中能够驻留的线程总数相当，但也有可能更小。只有在~GPU~满负荷工作的情况下，GPU~中的计算资源才能充分地发挥作用，从而获得较高的加速比。

因为我们的~CPU~程序中的计算是串行的，其性能基本上与数组长度无关，所以~GPU~程序相对于~CPU~程序的加速比在小~\verb"N"~的极限下几乎是正比于~\verb"N"~的。在大~\verb"N"~的极限下，GPU~程序相对于~CPU~程序的加速比接近饱和。总之，对于数据规模很小的问题，用~GPU~很难得到可观的加速。




\section{CUDA~中的数学函数库}

在前面的例子中，我们在核函数中使用了求平方根的数学函数。在~CUDA~数学库中，还有很多类似的数学函数，如幂函数、三角函数、指数函数、对数函数等。这些函数可以在如下网站查询：\url{http://docs.nvidia.com/cuda/cuda-math-api}。建议读者浏览该文档，了解~CUDA~的数学函数库都提供了哪些数学函数。这样，在需要使用时就容易想起来。

CUDA~数学库中的函数可以归纳如下：
\begin{enumerate}
    \item 单精度浮点数内建函数和数学函数（single precision intrinsics and math functions）。使用该类函数时不需要包含任何额外的头文件。
    \item 双精度浮点数内建函数和数学函数（double precision intrinsics and math functions）。使用该类函数时不需要包含任何额外的头文件。
    \item 半精度浮点数内建函数和数学函数（half precision intrinsics and math functions）。使用该类函数时需要包含头文件~\verb"<cuda_fp16.h>"。本书不涉及此类函数。
    \item 整数类型的内建函数（integer intrinsics）。使用该类函数时不需要包含任何额外的头文件。本书不涉及此类函数。
    \item 类型转换内建函数（type casting intrinsics）。使用该类函数时不需要包含任何额外的头文件。本书不涉及此类函数。
    \item 单指令-多数据内建函数（SIMD intrinsics）。使用该类函数时不需要包含任何额外的头文件。本书不涉及此类函数。
\end{enumerate}

本书将仅涉及单精度浮点数和双精度浮点数类型的数学函数和内建函数。其中数学函数（math functions）都是经过重载的。例如，求平方根的函数具有如下3种原型：
\begin{verbatim}
    double sqrt(double x);
    float sqrt(float x);
    float sqrtf(float x);
\end{verbatim}
所以，当~\verb"x"~是双精度浮点数时，我们只可以用~\verb"sqrt(x)"；当~\verb"x"~是单精度浮点数时，我们可以用~\verb"sqrt(x)"，也可以用~\verb"sqrtf(x)"。那么综合起来，我们可统一地用双精度函数的版本处理单精度浮点数和双精度浮点数类型的变量。

内建函数指的是一些准确度较低，但效率较高的函数。例如，有如下版本的求平方根的内建函数：
\begin{verbatim}
__device__​ float __fsqrt_rd (float  x); // round-down mode
__device__​ float __fsqrt_rn (float  x); // round-to-nearest-even mode
__device__​ float __fsqrt_ru (float  x); // round-up mode
__device__​ float __fsqrt_rz (float  x); // round-towards-zero mode
__device__​ double __fsqrt_rd (double  x); // round-down mode
__device__​ double __fsqrt_rn (double  x); // round-to-nearest-even mode
__device__​ double __fsqrt_ru (double  x); // round-up mode
__device__​ double __fsqrt_rz (double  x); // round-towards-zero mode   
\end{verbatim}

在开发~CUDA~程序时，浮点数精度的选择及数学函数和内建函数之间的选择都要视应用程序的要求而定。例如，在作者开发的分子动力学模拟程序~GPUMD（\url{https://github.com/brucefan1983/GPUMD}）中，绝大部分的代码使用了双精度浮点数，只在极个别的地方使用了单精度浮点数，而且没有使用内建函数；在作者开发的经验势拟合程序~GPUGA（\url{https://github.com/brucefan1983/GPUGA}）中，统一使用了单精度浮点数，而且使用了内建函数。之所以这样选择，是因为前者对计算精度要求较高，后者对计算精度要求较低。


