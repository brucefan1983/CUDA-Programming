\chapter{CUDA~程序的错误检测\label{chapter:error-check}}

和编写~C++~程序一样，编写~CUDA~程序时难免会出现各种各样的错误。有的错误在编译的过程中就可以被编译器捕捉，称为编译错误。有的错误在编译期间没有被发现，但在运行的时候出现，称为运行时刻的错误。一般来说，运行时刻的错误更难排错。本章讨论如何检测运行时刻的错误，包括使用一个检查~CUDA~运行时~API~函数返回值的宏函数及使用~CUDA-MEMCHECK~工具。

\section{一个检测~CUDA~运行时错误的宏函数}

在第~\ref{chapter:framework}~章，我们学习了一些~CUDA~运行时~API~函数，如分配设备内存的函数~\verb"cudaMalloc()"、释放设备内存的函数~\verb"cudaFree()"~及传输数据的函数~\verb"cudaMemcpy()"。所有~CUDA~运行时~API~函数都是以~\verb"cuda"为前缀的，而且都有一个类型为~\verb"cudaError_t"~的返回值，代表了一种错误信息。只有返回值为~\verb"cudaSuccess"~时才代表成功地调用了~API~函数。

\begin{lstlisting}[language=C++,caption={本书中使用的一个检测~CUDA~运行时错误的宏函数。},label={listing:error.cuh}]
#pragma once
#include <stdio.h>

#define CHECK(call)                                                     \
do                                                                      \
{                                                                       \
    const cudaError_t error_code = call;                                \
    if (error_code != cudaSuccess)                                      \
    {                                                                   \
        printf("CUDA Error:\n");                                        \
        printf("    File:       %s\n", __FILE__);                       \
        printf("    Line:       %d\n", __LINE__);                       \
        printf("    Error code: %d\n", error_code);                     \
        printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
        exit(1);                                                        \
    }                                                                   \
} while (0)
\end{lstlisting}

根据这样的规则，我们可以写出一个头文件（error.cuh），它包含一个检测~CUDA~运行时错误的宏函数（macro function），见~Listing~\ref{listing:error.cuh}。对该宏函数的解释如下：
\begin{itemize}
    \item 该文件开头一行的~\verb"#pragma once"~是一个预处理指令，其作用是确保当前文件在一个编译单元中不被重复包含。该预处理指令和如下复合的预处理指令作用相当，但更加简洁：
\begin{verbatim}
    #ifndef ERROR_CUH_
    #define ERROR_CUH_
        头文件中的内容（即上述文件中第 2-17 行的内容）
    #endif
\end{verbatim}
    \item 该宏函数的名称是~\verb"CHECK"，参数~\verb"call"~是一个~CUDA~运行时~API~函数。
    \item 在定义宏时，如果一行写不下，需要在行末写~\verb"\"，表示续行。
    \item 第~7~行定义了一个~\verb"cudaError_t"~类型的变量~\verb"error_code"，并初始化为函数~\verb"call"~的返回值。
    \item 第~8~行判断该变量的值是否为~\verb"cudaSuccess"。如果不是，在第~9-16~行报道相关文件、行数、错误代号及错误的文字描述并退出程序。\verb"cudaGetErrorString"~显然也是一个~CUDA~运行时~API~函数，作用是将错误代号转化为错误的文字描述。
\end{itemize}

在使用该宏函数时，只要将一个~CUDA~运行时~API~函数当作参数传入该宏函数即可。例如，如下宏函数的调用
\begin{verbatim}
    CHECK(cudaFree(d_x));
\end{verbatim}
将会被展开为~Listing \ref{listing:expansion}~所示的代码段。

\begin{lstlisting}[language=C++,caption={宏函数调用的展开。},label={listing:expansion}]
do
{
    const cudaError_t error_code = cudaFree(d_x);              
    if (error_code != cudaSuccess)                    
    {                                                 
        printf("CUDA Error:\n");                      
        printf("    File:       %s\n", __FILE__);     
        printf("    Line:       %d\n", __LINE__);     
        printf("    Error code: %d\n", error_code);   
        printf("    Error text: %s\n", cudaGetErrorString(error_code));          
        exit(1);                                      
    }
} while (0);
\end{lstlisting}

读者可能会问，宏函数的定义中为什么用了一个~\verb"do-while"~语句？不用该语句在大部分情况下也是可以的，但在某些情况下不安全（这里不对此展开讨论，感兴趣的读者可自行研究）。也可以不用宏函数，而用普通的函数，但此时必须将宏~\verb"__FILE__"~和~\verb"__LINE__"~传给该函数，这样用起来不如宏函数简洁。

\subsection{检查运行时~API~函数}

作为一个例子，我们将第~\ref{chapter:framework}~章的程序~\verb"add2wrong.cu"~中的~CUDA~运行时API函数都用宏函数~\verb"CHECK"~进行包装，得到~\verb"check1api.cu"，部分代码见~Listing \ref{listing:check1api.cu}。在该文件的开头，包含了上述头文件：
\begin{verbatim}
    #include "error.cuh"
\end{verbatim}
第~27-29~行对分配设备内存的函数进行了检查；第~30-31~行及第~37~行对数据传输的函数进行了检查；第~43-45~行对释放设备内存的函数进行了检查。用
\begin{verbatim}
    $ nvcc -arch=sm_75 check1api.cu
\end{verbatim}
编译该程序，然后运行得到的可执行文件，将得到如下输出：
\begin{verbatim}
    CUDA Error:
        File:       check1api.cu
        Line:       30
        Error code: 11
        Error text: invalid argument
\end{verbatim}
可见，宏函数正确地捕捉到了运行时刻的错误，告诉我们文件~\verb"check1api.cu"~的第~30~行代码中出现了非法的参数。非法参数指的是~\verb"cudaMemcpy"~函数的参数有问题，因为我们故意将~cudaMemcpyHostToDevice~写成了~cudaMemcpyDeviceToHost。可见，用了检查错误的宏函数之后，我们可以得到更有用的错误信息，而不仅仅是一个错误的运行结果。从这里开始，我们将坚持用这个宏函数包装大部分的~CUDA~运行时~API~函数。有一个例外是~cudaEventQuery~函数，因为它很有可能返回 cudaErrorNotReady，但又不代表程序出错了。

\begin{lstlisting}[language=C++,caption={本章程序~check1api.cu~中的部分代码。},label={listing:check1api.cu}]
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
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyDeviceToHost));

    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

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



\subsection{检查核函数}

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

