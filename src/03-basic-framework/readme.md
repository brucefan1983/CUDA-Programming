**unfinished...**

# Chapter 3 Basic framework of simple CUDA programs

## 3.1 An example: adding up two arrays

We consider a simple task: adding up to arrays. We first give a C++ program [add.cpp](https://github.com/brucefan1983/CUDA-Programming/tree/master/src/03-basic-framework/add.cpp) solving this problem.
```
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }

    add(x, y, z, N);
    check(z, N);

    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const double *x, const double *y, double *z, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}
\end{lstlisting}
```

The above program can be compiled by using `g++` or the `cl.exe`. Running the executable, we will see the following message on the screen,
```
No errors
``` 
which indicates that the result from the `add` function is correct. The reader should be able to understand this program, otherwise he/she needs to review some knowledge of C++ programming first.

\section{CUDA~程序的基本框架}

在现实的中、大型程序中，往往使用多个源文件，每个源文件又包含多个函数。本书第~\ref{chapter:md}~的例子就是这样。然而，在其他章节的例子中，我们只使用一个源文件，其中包含一个主函数和若干其他 函数（包括~C++~自定义函数和~CUDA~核函数）。在这种情况下，一个典型的~CUDA~程序的基本框架见~Listing \ref{listing:typical-cuda}。

\begin{lstlisting}[language=C++,caption={一个典型的~CUDA~程序的基本框架。},label={listing:typical-cuda}]
头文件包含
常量定义（或者宏定义）
C++ 自定义函数和 CUDA 核函数的声明（原型）
int main(void)
{
    分配主机与设备内存
    初始化主机中的数据
    将某些数据从主机复制到设备
    调用核函数在设备中进行计算
    将某些数据从设备复制到主机
    释放主机与设备内存
}
C++ 自定义函数和 CUDA 核函数的定义（实现）
\end{lstlisting}

在上述~CUDA~程序的基本框架中，有很多内容还没有介绍。但是，我们先把利用~CUDA~求数组之和的全部源代码列出来，之后再逐步讲解。Listing \ref{listing:add1.cu}~给出了除~\verb"check"~函数定义（该函数和前一个~C++~程序中的同名函数具有相同的定义）之外的全部源代码。

\begin{lstlisting}[language=C++,caption={本章程序~add1.cu~中的大部分内容。},label={listing:add1.cu}]
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z);
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
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = N / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

void __global__ add(const double *x, const double *y, double *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}
\end{lstlisting} 


用~\verb"nvcc"~编译该程序，并指定与~GeForce RTX 2070~对应的计算能力（读者可以选用自己所用~GPU~的计算能力）：
\begin{verbatim}
    $ nvcc -arch=sm_75 add1.cu
\end{verbatim}
将得到一个可执行文件~\verb"a.out"。运行该程序得到的输出应该与前面~C++~程序所得到的输出一样，说明得到了预期的结果。

值得注意的是，当使用较大的数据量时，网格大小往往很大。例如，本例中的网格大小为~$10^8/128=781250$。如果读者使用~CUDA 8.0，而在用~\verb"nvcc"~编译程序时又忘了指定一个计算能力，那就会根据默认的~2.0~的计算能力编译程序。对于该计算能力，网格大小在~\verb"x"~方向的上限为~65535，小于本例中所使用的值。这将导致程序无法正确地执行。这是初学者需要特别注意的一个问题。下面对该程序进行详细的讲解。

\subsection{隐形的设备初始化}

在CUDA运行时API中，没有明显地初始化设备（即GPU）的函数。在第一次调用一个和设备管理及版本查询功能无关的运行时API函数时，设备将自动地初始化。

\subsection{设备内存的分配与释放}

在上述程序中，我们首先在主机中定义了3个数组并进行了初始化。这与之前~C++~版本的相应部分是一样的。接着，第~25-28~行在设备中也定义了3个数组并分配了内存（显存）。第~25~行就是定义3个双精度类型变量的指针。如果不看后面的代码，我们并不知道这3个指针会指向哪些内存区域。只有通过第~26-28~行的~\verb"cudaMalloc()"~函数才能确定它们将指向设备中的内存，而不是主机中的内存。该函数是一个~CUDA~运行时~API~函数。所有~CUDA~运行时~API~函数都以~\verb"cuda"~开头。本书仅涉及极少数的~CUDA~运行时~API~函数。完整的列表见如下网页（一个几百页的手册）：\url{https://docs.nvidia.com/cuda/cuda-runtime-api}。

正如在~C++~中可由~\verb"malloc()"~函数动态分配内存，在~CUDA~中，设备内存的动态分配可由~\verb"cudaMalloc()"~函数实现。该函数的原型如下：
\begin{verbatim}
    cudaError_t cudaMalloc(void **address, size_t size);			
\end{verbatim}
其中：
\begin{itemize}
\item 第一个参数~\verb"address"~是待分配设备内存的指针。注意：因为内存（地址）本身就是一个指针，所以待分配设备内存的指针就是指针的指针，即双重指针。
\item 第二个参数~\verb"size"~是待分配内存的字节数。
\item 返回值是一个错误代号。如果调用成功，返回~\verb"cudaSuccess"，否则返回一个代表某种错误的代号（下一章会进一步讨论）。
\end{itemize}
该函数为某个变量分配~\verb"size" 字节的线性内存（linear memory）。初学者不必深究什么是线性内存，也暂时不用关心该函数的返回值。在第26-28行，我们忽略了函数~\verb"cudaMalloc()"~的返回值。这几行代码用到的参数~\verb"M"~是所分配内存的字节数，即~\verb"sizeof(double) * N"。注意：虽然在很多情况下~\verb"sizeof(double)"~等于~8，但用~\verb"sizeof(double)" 是更加通用、安全的做法。

调用函数~\verb"cudaMalloc()"~时传入的第一个参数~\verb"(void **)&d_x"~稍难理解。首先，我们知道~\verb"d_x"~是一个~\verb"double"~类型的指针，那么它的地址~\verb"&d_x"~就是~\verb"double"~类型的双重指针。而~\verb"(void **)"~是一个强制类型转换操作，将一个某种类型的双重指针转换为一个~\verb"void"~类型的双重指针。这种类型转换可以不明确地写出来，即对函数~\verb"cudaMalloc()"~的调用可以简写为
\begin{verbatim}
    cudaMalloc(&d_x, M); 			
\end{verbatim}
读者可以自行试一试。

读者也许会问，\verb"cudaMalloc()"~函数为什么需要一个双重指针作为变量呢？这是因为（以第~26~行为例），该函数的功能是改变指针~\verb"d_x"~本身的值（将一个指针赋值给~\verb"d_x"），而不是改变~\verb"d_x"~所指内存缓冲区中的变量值。在这种情况下，必须将~\verb"d_x"~的地址~\verb"&d_x"~传给函数~\verb"cudaMalloc()"~才能达到此效果。这是~C++~编程中非常重要的一点。如果读者对指针的概念比较模糊，请务必阅读相关资料，查漏补缺。从另一个角度来说，函数~\verb"cudaMalloc()"~要求用传双重指针的方式改变一个指针的值，而不是直接返回一个指针，是因为该函数已经将返回值用于返回错误代号，而~C++~又不支持多个返回值。

总之，用~\verb"cudaMalloc()" 函数可以为不同类型的指针变量分配设备内存。注意，为了区分主机和设备中的变量，我们（遵循~CUDA~编程的传统）用~\verb"d_"~作为所有设备变量的前缀，而用~\verb"h_"~作为对应主机变量的前缀。

正如用~\verb"malloc()"~函数分配的主机内存需要用~\verb"free()"~函数释放一样，用~\verb"cudaMalloc()"~函数分配的设备内存需要用~\verb"cudaFree()"~函数释放。该函数的原型为
\begin{verbatim}
    cudaError_t cudaFree(void* address);   
\end{verbatim}
这里，参数~\verb"address"~就是待释放的设备内存变量（不是双重指针）。返回值是一个错误代号。如果调用成功，返回~\verb"cudaSuccess"。

主机内存也可由~C++~中的~\verb"new"~运算符动态分配，并由~\verb"delete"~运算符释放。读者可以将程序~\verb"add1.cu"~中的~\verb"malloc()"~和~\verb"free()"~语句分别换成用~\verb"new"~和~\verb"delete"~实现的等价的语句，看看是否能正确地编译、运行。

在分配与释放各种内存时，相应的操作一定要两两配对，否则将有可能出现内存错误。将程序~\verb"add1.cu"~中的~\verb"cudaFree()"~改成~\verb"free()"，虽然能够正确地编译，而且能够在屏幕打印出~\verb"No errors"~的结果，但在程序退出之前，还是会出现所谓的段错误（segmentation fault）。读者可以自行试一试。主动尝试错误是编程学习中非常重要的技巧，因为通过它可以熟悉各种编译和运行错误，提高排错能力。

从计算能力~2.0~开始，CUDA~还允许在核函数内部用~\verb"malloc()"~和~\verb"free()"~动态地分配与释放一定数量的全局内存。一般情况下，这样容易导致较差的程序性能，不建议使用。如果发现有这样的需求，可能需要思考如何重构算法。

\subsection{主机与设备之间数据的传递}

在分配了设备内存之后，就可以将某些数据从主机传递到设备中去了。第~29-30~行将主机中存放在~\verb"h_x"~和~\verb"h_y"~中的数据复制到设备中的相应变量~\verb"d_x"~和~\verb"d_y"~所指向的缓冲区中去。这里用到了~CUDA~运行时~API~函数~\verb"cudaMemcpy()"，其原型是：
\begin{verbatim}
    cudaError_t cudaMemcpy
    ( 	
        void                *dst,
        const void          *src,
        size_t              count,
        enum cudaMemcpyKind kind	
    );
\end{verbatim}
其中：
\begin{itemize}
\item 第一个参数~\verb"dst"~是目标地址。
\item 第二个参数~\verb"src"~是源地址。
\item 第三个参数~\verb"count"~是复制数据的字节数。
\item 第四个参数~\verb"kind"~一个枚举类型的变量，标志数据传递方向。它只能取如下几个值：
\begin{itemize}
\item \verb"cudaMemcpyHostToHost"，表示从主机复制到主机。
\item \verb"cudaMemcpyHostToDevice"，表示从主机复制到设备。
\item \verb"cudaMemcpyDeviceToHost"，表示从设备复制到主机。
\item \verb"cudaMemcpyDeviceToDevice"，表示从设备复制到设备。
\item \verb"cudaMemcpyDefault"，表示根据指针~\verb"dst"~和~\verb"src"~所指地址自动判断数据传输的方向。这要求系统具有统一虚拟寻址（unified virtual addressing）的功能（要求~64~位的主机）。CUDA~正在逐步放弃对~32~位主机的支持，故一般情况下用该选项自动确定数据传输方向是没有问题的。至于是明确地指定传输方向更好，还是利用自动判断更好，则是一个仁者见仁、智者见智的问题。
\end{itemize}
\item 返回值是一个错误代号。如果调用成功，返回~\verb"cudaSuccess"。
\item 该函数的作用是将一定字节数的数据从源地址所指缓冲区复制到目标地址所指缓冲区。
\end{itemize}


我们回头看程序的第~29~行。它的作用就是将~\verb"h_x"~指向的主机内存中~\verb"M"~字节的数据复制到~\verb"d_x"~指向的设备内存中去。因为这里的源地址是主机中的内存，目标地址是设备中的内存，所以第四个参数必须是~\verb"cudaMemcpyHostToDevice"~或~\verb"cudaMemcpyDefault"，否则将导致错误。

类似地，在调用核函数进行计算，得到需要的数据之后，我们需要将设备中的数据复制到主机，这正是第~36~行的代码所做的事情。该行代码的作用就是将~\verb"d_z"~指向的设备内存中~\verb"M"~字节的数据复制到~\verb"h_z"~指向的主机内存中去。因为这里的源地址是设备中的内存，目标地址是主机中的内存，所以第四个参数必须是~\verb"cudaMemcpyDeviceToHost"~或~\verb"cudaMemcpyDefault"，否则将导致错误。

在本章的程序~\verb"add2wrong.cu"~中，作者故意将第~29-30~行的传输方向参数写成了~\verb"cudaMemcpyDeviceToHost"。请读者编译、运行该程序，看看会得到什么结果。

\subsection{核函数中数据与线程的对应}

将有关的数据从主机传至设备之后，就可以调用核函数在设备中进行计算了。第~32-34~行确定了核函数的执行配置：使用具有~128~个线程的一维线程块，一共有~$10^8/128$~个这样的线程块。仔细比较程序~\verb"add.cpp"~中的主机端函数（第~35-41~行）和程序~\verb"add1.cu"~中的设备端函数（第~48-52~行），可以看出，将主机中的函数改为设备中的核函数是非常简单的：基本上就是去掉一层循环。在主机函数中，我们需要依次对数组的每一个元素进行操作，所以需要使用一个循环。在设备的核函数中，我们用“单指令-多线程”的方式编写代码，故可去掉该循环，只需将数组元素指标与线程指标一一对应即可。

例如，在上述核函数中，使用了语句
\begin{verbatim}
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
\end{verbatim}
来确定对应方式。赋值号右边只出现标记线程的内建变量，左边的~\verb"n"~是后面代码中将要用到的数组元素指标。在这种情况下，第~0~号线程块中的~\verb"blockDim.x"~个线程对应于第0个到第~\verb"blockDim.x-1"~个数组元素，第~1~号线程块中的~\verb"blockDim.x"~个线程对应于第~\verb"blockDim.x"~个到第~\verb"2*blockDim.x-1"~个数组元素，第~2~号线程块中的~\verb"blockDim.x"~个线程对应于第~\verb"2*blockDim.x"~个到第~\verb"3*blockDim.x-1"~个数组元素，依此类推。这里的~\verb"blockDim.x"~等于执行配置中指定的（一维）线程块大小。核函数中定义的线程数目与数组元素数目一样，都是~$10^8$。在将线程指标与数据指标一一对应之后，就可以对数组元素进行操作了。该操作的语句为
\begin{verbatim}
    z[n] = x[n] + y[n];
\end{verbatim}
在主机函数中和核函数中是一样的。通常，在写出一个主机端的函数后，翻译成核函数是非常直接的。最后，值得一提的是，在调试程序时，也可以仅仅使用一个线程。为此，可以将核函数中的代码改成对应主机函数中的代码（即有~for~循环的代码），然后用执行配置~\verb"<<<1, 1>>>"~调用核函数。

\subsection{核函数的要求}

核函数无疑是~CUDA~编程中最重要的方面。我们这里列出编写核函数时要注意的几点：
\begin{itemize}
\item 核函数的返回类型必须是~\verb"void"。所以，在核函数中可以用~\verb"return"~关键字，但不可返回任何值。
\item 必须使用限定符~\verb"__global__"。也可以加上一些其他C++中的限定符，如~\verb"static"。限定符的次序可任意。
\item 函数名无特殊要求，而且支持~C++~ 中的重载（overload），即可以用同一个函数名表示具有不同参数列表的函数。
\item 不支持可变数量的参数列表，即参数的个数必须确定。
\item 可以向核函数传递非指针变量（如例子中的~\verb"int N"），其内容对每个线程可见。
\item 除非使用统一内存编程机制（将在第~\ref{chapter:unified-memory}~章介绍），否则传给核函数的数组（指针）必须指向设备内存。
\item 核函数不可成为一个类的成员。通常的做法是用一个包装函数调用核函数，而将包装函数定义为类的成员。
\item 在计算能力~3.5~之前，核函数之间不能相互调用。从计算能力~3.5~开始，引入了动态并行（dynamic parallelism）机制，在核函数内部可以调用其他核函数，甚至可以调用自己（这样的函数称为递归函数）。但本书不讨论动态并行，感兴趣的读者请参考《CUDA C++ Programming Guide》的附录~D。
\item 无论是从主机调用，还是从设备调用，核函数都是在设备中执行。调用核函数时必须指定执行配置，即三括号和它里面的参数。在本例中，选取的线程块大小为~128，网格大小为数组元素个数除以线程块大小，即~$10^8/128=781250$。
\end{itemize}

\subsection{核函数中~if~语句的必要性}

前面的核函数根本没有使用参数~\verb"N"。当~\verb"N"~是~\verb"blockDim.x"（即~\verb"block_size"）的整数倍时，不会引起问题，因为核函数中的线程数目刚好等于数组元素的个数。然而，当~\verb"N"~不是~\verb"blockDim.x"~的整数倍时，就有可能引发错误。

我们将~\verb"N"~改为~$10^8+1$，而且依然取~\verb"block_size"~等于~128。此时，我们首先面临的一个问题就是，
\verb"grid_size" 应该取多大？用~\verb"N" 除以~\verb"block_size"，商为~$781250$，余数为~1。显然，我们不能取
~\verb"grid_size"为~$781250$，因为这样只能定义~$10^8$~个线程，在用一个线程对应一个数组元素的方案下无法处理剩下的~1 个元素。实际上，我们可以将~\verb"grid_size" 取为~$781251$，使得定义的线程数为~$10^8+128$。虽然定义的总线程数多于元素个数，但我们可以通过条件语句规避不需要的线程操作。据此，我们可以写出如~Listing \ref{listing:add3if.cu}~所示的核函数。此时，在主机中调用该核函时所用的~\verb"grid_size"~为
\begin{verbatim}
    int grid_size  = (N - 1) / block_size + 1;
\end{verbatim}
或者
\begin{verbatim}
    int grid_size  = (N + block_size - 1) / block_size;
\end{verbatim}
以上两个语句都等价于下述语句：
\begin{verbatim}
    int grid_size = (N % block_size == 0) 
                  ? (N / block_size) 
                  : (N / block_size + 1);
\end{verbatim}
因为此时线程数（$10^8+128$）多于数组元素个数（$10^8+1$），所以如果去掉~\verb"if" 语句，则会出现非法的设备内存操作，可能导致不可预料的错误。这是在~CUDA~编程中一定要避免的。另外，虽然核函数不允许有返回值，但还是可以使用~return~语句。上述核函数中的代码也可以写为如下等价的形式：
\begin{verbatim}
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= N) return;
    z[n] = x[n] + y[n];
\end{verbatim}


\begin{lstlisting}[language=C++,caption={本章程序~add3if.cu~中的核函数定义。},label={listing:add3if.cu}]
void __global__ add(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}
\end{lstlisting}



\section{自定义设备函数}

核函数可以调用不带执行配置的自定义函数，这样的自定义函数称为设备函数（device function）。它是在设备中执行，并在设备中被调用的。与之相比，核函数是在设备中执行，但在主机端被调用的。现在也支持在一个核函数中调用其他核函数，甚至该核函数本身，但本书不涉及这方面的内容。设备函数的定义与使用涉及CUDA中函数执行空间标识符的概念。我们先对此进行介绍，然后以数组相加的程序为例展示设备函数的定义与调用。

\subsection{函数执行空间标识符}

在~CUDA~程序中，由以下标识符确定一个函数在哪里被调用以及在哪里执行：
\begin{itemize}
    \item 用~\verb"__global__"~修饰的函数称为核函数，一般由主机调用，在设备中执行。如果使用动态并行，则也可以在核函数中调用自己或其他核函数。
    \item 用~\verb"__device__"~修饰的函数叫称为备函数，只能被核函数或其他设备函数调用，在设备中执行。
    \item 用~\verb"__host__"~修饰的函数就是主机端的普通~C++~函数，在主机中被调用，在主机中执行。对于主机端的函数，该修饰符可省略。之所以提供这样一个修饰符，是因为有时可以用~\verb"__host__"~和~\verb"__device__"~同时修饰一个函数，使得该函数既是一个~C++~中的普通函数，又是一个设备函数。这样做可以减少冗余代码。编译器将针对主机和设备分别编译该函数。
    \item 不能同时用~\verb"__device__"~和~\verb"__global__"~修饰一个函数，即不能将一个函数同时定义为设备函数和核函数。
    \item 也不能同时用~\verb"__host__"~和~\verb"__global__"~修饰一个函数，即不能将一个函数同时定义为主机函数和核函数。
    \item 编译器决定把设备函数当作内联函数（inline function）或非内联函数，但可以用修饰符~\verb"__noinline__"~建议一个设备函数为非内联函数（编译器不一定接受），也可以用修饰符~\verb"__forceinline__"~建议一个设备函数为内联函数。
\end{itemize}

\subsection{例子：为数组相加的核函数定义一个设备函数}

Listing~\ref{listing:add4device.cu}~给出了3个版本的设备函数及调用它们的核函数。这3个版本的设备函数分别利用返回值、指针和引用（reference）返回结果。这里涉及的语法和~C++~中函数定义与调用的语法是一致的，故不再多做解释。这几种定义设备函数的方式不会导致程序性能的差别，读者可选择自己喜欢的风格。

\begin{lstlisting}[language=C++,caption={本章程序~add4device.cu~中的核函数和设备函数的定义。},label={listing:add4device.cu}]
// 版本一：有返回值的设备函数
double __device__ add1_device(const double x, const double y)
{
    return (x + y);
}

void __global__ add1(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = add1_device(x[n], y[n]);
    }
}

// 版本二：用指针的设备函数
void __device__ add2_device(const double x, const double y, double *z)
{
    *z = x + y;
}

void __global__ add2(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        add2_device(x[n], y[n], &z[n]);
    }
}

// 版本三：用引用（reference）的设备函数
void __device__ add3_device(const double x, const double y, double &z)
{
    z = x + y;
}

void __global__ add3(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        add3_device(x[n], y[n], z[n]);
    }
}
\end{lstlisting}   
