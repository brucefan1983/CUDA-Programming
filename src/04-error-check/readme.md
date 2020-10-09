# Chapter 4: Error checking in CUDA programs

In this chapter, we show how to check CUDA runtime API functions and CUDA kernels.



## 4.1 A macro function checking CUDA runtime API functions

In the last chapter, we have learned some CUDA runtime API functions, such as `cudaMalloc`, `cudaFree`, and `cudaMemcpy`. All but very few CUDA runtime API functions return a value, which indicates a type of error when it is not `cudaSuccess`. Based on this, we can write a macro function which can check this return value for a CUDA runtime API function and report a meaningful error message when the API function is not successfully called. The macro function is presented in [error.cuh](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/04-error-check/error.cuh), as given below:

```c++
#pragma once
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

```



### 4.1.1 Checking CUDA runtime API functions using the macro function

As an example, we check all the CUDA API functions in the [add2wrong.cu](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/03-basic-framework/add2wrong.cu) program of Chapter 3, obtaining the [check1api.cu](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/04-error-check/check1api.cu) program of this chapter. We can compile this program using 

```shell
$ nvcc -arch=sm_75 check1api.cu
```

Running the executable, we will get the following output:

```shell
    CUDA Error:
        File:       check1api.cu
        Line:       30
        Error code: 11
        Error text: invalid argument
```

We see that the macro function captured the error, telling us that there is invalid argument in line 30 of the source file. Here, the invalid argument is the last one, `cudaMemcpyDeviceToHost`, which should be `cudaMemcpyHostToDevice`. 

### 4.1.2 Checking CUDA kernels using the macro function

**I am up to here...**

用上述方法不能捕捉调用核函数的相关错误，因为核函数不返回任何值（回顾一下，核函必须用~\verb"void"~修饰）。有一个方法可以捕捉调用核函数可能发生的错误，即在调用核函数之后加上如下两个语句：
\begin{verbatim}
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
\end{verbatim}
其中，第一个语句的作用是捕捉第二个语句之前的最后一个错误，第二个语句的作用是同步主机与设备。之所以要同步主机与设备，是因为核函数的调用是异步的，即主机发出调用核函数的命令后会立即执行后面的语句，不会等待核函数执行完毕。关于核函数调用的异步性，我们将在第~\ref{chapter:cuda-stream}~章中详细讨论。在这之前，我们无须对此深究。需要注意的是，上述同步函数是比较耗时的，如果在程序的较内层循环调用的话，很可能会严重地降低程序的性能。所以，一般不在程序的较内层循环调用上述同步函数。只要在核函数的调用之后还有对其他任何能返回错误值的~API~函数进行同步调用，都能够触发主机与设备的同步并捕捉到核函数调用中可能发生的错误。

为了展示对核函数调用的检查，我们在第~\ref{chapter:framework}~章的程序~\verb"add1.cu"~的基础上写一个有错误的程序~\verb"check2kernel.cu"，见~Listing \ref{listing:check2kernel.cu}。在第~\ref{chapter:thread}~章我们提到过，线程块大小的最大值是~1024（这对从开普勒到图灵的所有架构都成立）。假如我们不小心将核函数执行配置中的线程块大小写成了~1280，该核函数将不能被成功地调用。第~36~行的代码成功地捕获了该错误，告诉我们程序中核函数的执行配置参数有误：
\begin{verbatim}
    CUDA Error:
        File:       check4kernel.cu
        Line:       36
        Error code: 9
        Error text: invalid configuration argument
\end{verbatim}
如果不用宏函数检查（即去掉第36-37行的代码），则很难知道错误的原因，只能看到程序给出~\verb"Has errors"~的输出结果（因为执行配置错误，核函数无法正确执行，从而无法计算出正确的结果）。

\begin{lstlisting}[language=C++,caption={本章检查核函数调用的示例程序中的部分代码。},label={listing:check2kernel.cu}]
#include "error.cuh"
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }
    
    double *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMalloc((void **)&d_y, M));
    CHECK(cudaMalloc((void **)&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));
    
    const int block_size = 1280;
    const int grid_size = (N + block_size - 1) / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    check(h_z, N);
    
    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}
\end{lstlisting}

在该例子中，去掉第~37~行对同步函数的调用也能成功地捕捉到上述错误信息。这是因为，第~39~行的数据传输函数起到了一种隐式的（implicit）同步主机与设备的作用。在一般情况下，如果要获得精确的出错位置，还是需要显式的（explicit）同步。例如，调用~\verb"cudaDeviceSynchronize"~函数，或者临时将环境变量~\verb"CUDA_LAUNCH_BLOCKING"~的值设置为~1：
\begin{verbatim}
    $ export CUDA_LAUNCH_BLOCKING=1        
\end{verbatim}
这样设置之后，所有核函数的调用都将不再是异步的，而是同步的。也就是说，主机调用一个核函数之后，必须等待核函数执行完毕，才能往下走。这样的设置一般来说仅适用于调试程序，因为它会影响程序的性能。

\section{用~CUDA-MEMCHECK~检查内存错误}
CUDA~提供了名为~\verb"CUDA-MEMCHECK"~的工具集，具体包括~\verb"memcheck"、~\verb"racecheck"、~\verb"initcheck"、~\verb"synccheck"~共4个工具。它们可由可执行文件~\verb"cuda-memcheck"~调用：
\begin{verbatim}
   $ cuda-memcheck --tool memcheck [options] app_name [options] 
   $ cuda-memcheck --tool racecheck [options] app_name [options] 
   $ cuda-memcheck --tool initcheck [options] app_name [options]
   $ cuda-memcheck --tool synccheck [options] app_name [options]
\end{verbatim}
对于~\verb"memcheck"~工具，可以简化为
\begin{verbatim}
    $ cuda-memcheck [options] app_name [options]
\end{verbatim}

我们这里只给出一个使用~\verb"memcheck"~工具的例子。如果将第~\ref{chapter:framework}~章的文件~\verb"add3if.cu"~中的~\verb"if"~语句去掉，编译后用
\begin{verbatim}
    $ cuda-memcheck ./a.out
\end{verbatim}
运行程序，可得到一大串输出，其中最后一行为（读者得到的数字可能不一定是下面的~36）
\begin{verbatim}
    ========= ERROR SUMMARY: 36 error
\end{verbatim}
这说明程序有内存错误，与之前的讨论一致。将~\verb"if"~语句加上，编译后再用
\begin{verbatim}
    $ cuda-memcheck ./a.out
\end{verbatim}
运行，将得到简单的输出，其中最后一行为
\begin{verbatim}
    ========= ERROR SUMMARY: 0 errors
\end{verbatim}

在开发程序时，经常用~\verb"CUDA-MEMCHECK"~工具集检测内存错误是一个好的习惯。关于~\verb"CUDA-MEMCHECK"~的更多内容，参见~\url{https://docs.nvidia.com/cuda/cuda-memcheck}。最后要强调的是，最有效的防止出错的办法就是认真地写代码，并在写好之后认真地检查。

