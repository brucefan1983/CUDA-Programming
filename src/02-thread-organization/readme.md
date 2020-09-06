**Unfinished...**

# Chapter 2: Thread Organization in CUDA

We start with the simplest CUDA program: printing a `Hello World` string from the GPU.

## 2.1 A `Hello World` Program in C++

**To master CUDA C++, one must first master C++**, but we still begin with one of the simplest C++ program: printing a `Hello World` message to the console (screen).

To develop a simple C++ program, one can follow the following steps:
* Write the source code using a text editor (such as `gedit`; you can choose whatever you like).
* Use a compiler to compile the source code to obtain an object file and then use a linker to link the object file and some standard object files to obtain an executable. The compiling and linking processes are usually done with a single command and we will simply call it a compiling process. 
* Run the executable.

Let us first write the following program in a source file named `hello.cpp` (https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello.cpp).
```
#include <stdio.h>

int main(void)
{
    printf("Hello World!\n");
    return 0;
}
```

In Linux, we can use the following command to compile it:
```
$ g++ hello.cpp
```
This will generate an executable named `a.out` in the current directory. One can run this executable by typing
```
$ ./a.out
```
Then, one can see the following message printed in the console (screen):
```
Hello World!
```
One can also specify a name for the executable, e.g.,
```
$ g++ hello.cpp -o hello
```
This will generate an executable named `hello` in the current directory. One can run this executable by typing
```
$ ./hello
```

In Windows with the MSVC compiler `cl.exe`, one can compile the program using the following command in a command prompt:
```
$ cl.exe hello.cpp
```
This will generate an executable named `hello.exe`. It can be run using the following command
```
hello.exe
```

## A `Hello World` Program in CUDA

**I am up to here...**

在复习了C++语言中的Hello World程序之后，我们接着介绍CUDA中的Hello World程序。

### A CUDA program containing host functions only

其实，我们已经写好了一个CUDA中的Hello World程序。这是因为，CUDA 程序的编译器驱动（compiler driver）\verb"nvcc"支持编译纯粹的C++代码。一般来说，一个标准的CUDA程序中既有纯粹的C++代码，也有不属于C++的真正的CUDA代码。CUDA程序的编译器驱动\verb"nvcc"在编译一个CUDA程序时，会将纯粹的C++代码交给C++的编译器（如前面提到的\verb"g++"或\verb"cl"）去处理，它自己则负责编译剩下的部分。CUDA程序源文件的后缀名默认是\verb".cu"，所以我们可以将上面写好的源文件更名为\verb"hello1.cu"，然后用\verb"nvcc"编译：
\begin{verbatim}
    $ nvcc hello1.cu
\end{verbatim}
编译好之后即可运行。运行结果与C++程序的运行结果一样。关于CUDA程序的编译过程，将在本章最后一节及后续的某些章节详细讨论，现在只要知道可以用\verb"nvcc"编译CUDA程序即可。

### A CUDA program containing a CUDA kernel

虽然上面的第一个版本是由CUDA的编译器编译的，但程序中根本没有使用GPU。下面来介绍一个使用GPU的Hello World程序。

首先，我们要知道，GPU只是一个设备，要它工作的话还需要有一个主机给它下达命令。这个主机就是CPU。所以，一个真正利用了GPU的CUDA程序既有主机代码（在程序hello1.cu中的所有代码都是主机代码），也有设备代码（可以理解为需要设备执行的代码）。主机对设备的调用是通过核函数（kernel function）来实现的。所以，一个典型的、简单的CUDA程序的结构具有下面的形式：

\begin{verbatim}
    int main(void)
    {
        主机代码
　      核函数的调用
　      主机代码
        return 0;
    }
\end{verbatim}

CUDA中的核函数与C++中的函数是类似的，但一个显著的差别是：它必须被限定词（qualifier） \verb"__global__" 修饰。其中\verb"global"前后是双下划线。另外，核函数的返回类型必须是空类型，即\verb"void"。这两个要求读者先记住即可。关于核函数 的更多细节，以后再逐步深入介绍。遵循这两个要求，我们先写一个打印字符串的核函数：
\begin{verbatim}
    __global__ void hello_from_gpu()
    {
        printf("Hello World from the GPU!\n");
    }    
\end{verbatim}
限定符\verb"__global__"和\verb"void"的次序可随意。也就是说，上述核函数也可以写为如下形式：
\begin{verbatim}
    void __global__ hello_from_gpu()
    {
        printf("Hello World from the GPU!\n");
    }
\end{verbatim}

就像C++语言中的函数要被调用才能发挥作用一样，这个核函数 也要被调用才能发挥作用。下面，我们就写一个主函数来调用这个核函数，得到如Listing \ref{listing:hello2.cu} 所示的完整CUDA程序。我们可以用如下命令编译：
\begin{verbatim}
    $ nvcc hello2.cu
\end{verbatim}
然后运行得到的可执行文件就可从屏幕上看到如下输出：
\begin{verbatim}
    Hello World from the GPU!
\end{verbatim}

\begin{lstlisting}[language=C++,caption={本章程序hello2.cu中的内容。},label={listing:hello2.cu}]
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
\end{lstlisting}



上述程序有3个地方需要进一步解释：
\begin{itemize}
\item 先看看调用核函数 的格式：
\begin{verbatim}
    hello_from_gpu<<<1, 1>>>();
\end{verbatim}
这个调用格式与普通的C++函数的调用格式是有区别的。我们看到，在函数名\verb"hello_from_gpu"和括号\verb"()"之间有一对三括号\verb"<<<1, 1>>>"，里面还有用逗号隔开的两个数字。调用核函数时为什么需要这对三括号里面的信息呢？这是因为，一块GPU中有很多（例如，Tesla V100中有5120个）计算核心，从而可以支持很多线程（thread）。主机在调用一个核函数 时，必须指明需要在设备中指派多少个线程，不然设备不知道如何工作。三括号中的数就是用来指明核函数 中的线程数目以及排列情况的。核函数中的线程常组织为若干线程块（thread block）：三括号中的第一个数字可以看作线程块的个数，第二个数字可以看作每个线程块中的线程数。一个核函数的全部线程块构成一个网格（grid），而线程块的个数就记为网格大小（grid size）。每个线程块中含有同样数目的线程，该数目
称为线程块大小（block size）。所以，核函数中总的线程数就等于网格大小乘以线程块大小，而三括号中的两个数字分别就是网格大小和线程块大小，即\verb"<<<网格大小, 线程块大小>>>"。所以，在上述程序中，主机只指派了设备的一个线程，网格大小和线程块大小都是1，即$1\times 1 = 1$。
\item 核函数中的\verb"printf()"函数的使用方式和C++库（或者说C++从C中继承的库）中的\verb"printf()"函数的使用方式基本上是一样的。而且在核函数中使用\verb"printf()"函数时也需要包含头文件\verb"<stdio.h>"（也可以写成\verb"<cstdio>"）。需要注意的是，核函数中不支持C++的iostream（读者可亲自测试）。
\item 我们注意到，在调用核函数之后，有如下一行语句：
\begin{verbatim}
    cudaDeviceSynchronize();
\end{verbatim}
这行语句调用了一个CUDA的运行时API函数。去掉这个函数就打印不出字符串了（请读者亲自尝试）。这是因为调用输出函数时，输出流是先存放在缓冲区的，而缓冲区不会自动刷新。只有程序遇到某种同步操作时缓冲区才会刷新。函数\verb"cudaDeviceSynchronize"的作用是同步主机与设备，所以能够促使缓冲区刷新。读者现在不需要弄明白这个函数到底是什么，因为我们这里的主要目的是介绍CUDA中的线程组织。
\end{itemize}

## Thread organization in CUDA 

## A CUDA kernel using multiple threads

核函数中允许指派很多线程，这是一个必然的特征。这是因为，一个GPU往往有几千个计算核心，而总的线程数必须至少等于计算核心数时才有可能充分利用GPU中的全部计算资源。实际上，总的线程数大于计算核心数时才能更充分地利用GPU中的计算资源，因为这会让计算和内存访问之间及不同的计算之间合理地重叠，从而减小计算核心空闲的时间。

所以，根据需要，在调用核函数 时可以指定使用多个线程。Listing \ref{listing:hello3.cu}所示程序在调用核函数
\verb"hello_from_gpu"时指定了一个含有两个线程块的网格，而且每个线程块的大小是4。


\begin{lstlisting}[language=C++,caption={本章程序hello3.cu中的内容。},label={listing:hello3.cu}]
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
\end{lstlisting}

因为网格大小是2，线程块大小是4，故总的线程数是$2\times4=8$。 也就是说，该程序中的核函数 调用将指派8个线程。核函数中代码的执行方式是“单指令-多线程”，即每一个线程都执行同一串指令。既然核函数中的指令是打印一个字符串，那么编译、运行上述程序，将在屏幕打印如下8行同样的文字：
\begin{verbatim}
    Hello World from the GPU!
\end{verbatim}
其中，每一行对应一个指派的线程。读者也许要问，每一行分别是哪一个线程输出的呢？下面就来讨论这个问题。

## Using thread indices in a CUDA kernel

通过前面的介绍，我们知道，可以为一个核 函数指派多个线程，而这些线程的组织结构是由执行配置（execution configuration）
\begin{verbatim}
    <<<grid_size, block_size>>>
\end{verbatim}
决定的。这里的\verb"grid_size"（网格大小）和\verb"block_size" （线程块大小）一般来说是一个结构体类型的变量，但也可以是一个普通的整型变量。我们先考虑简单的整型变量，稍后再介绍更一般的情形。这两个整型变量的乘积就是被调用核函数中总的线程数。

我们强调过，本书不关心古老的特斯拉架构和费米架构。从开普勒架构开始，最大允许的线程块大小是1024，而最大允许的网格大小是$2^{31}-1$（针对这里的一维网格来说；后面介绍的多维网格能够定义更多的线程块）。 所以，用上述简单的执行配置时最多可以指派大约两万亿个线程。这通常是远大于一般的编程问题中常用的线程数目的。一般来说，只要线程数比GPU 中的计算核心数（几百至几千个）多几倍时，就有可能充分地利用GPU中的全部计算资源。总之，一个核函数允许指派的线程数目是巨大的，能够满足几乎所有应用程序的要求。需要指出的是，一个核函数中虽然可以指派如此巨大数目的线程数，但在执行时能够同时活跃（不活跃的线程处于等待状态）的线程数是由硬件（主要是CUDA 核心数）和软件（即核函数中的代码）决定的。

每个线程在核函数中都有一个唯一的身份标识。由于我们用两个参数指定了线程数目，那么自然地，每个线程的身份可由两个参数确定。在核函数内部，程序是知道执行配置参数\verb"grid_size" 和\verb"block_size" 的值的。这两个值分别保存于如下两个内建变量（built-in variable）：
\begin{itemize}
\item \verb"gridDim.x"：该变量的数值等于执行配置中变量\verb"grid_size" 的数值。
\item \verb"blockDim.x"：该变量的数值等于执行配置中变量\verb"block_size" 的数值。
\end{itemize}
类似地，在核函数中预定义了如下标识线程的内建变量：
\begin{itemize}
\item \verb"blockIdx.x"：该变量指定一个线程在一个网格中的线程块指标，其取值范围是从0 到\verb"gridDim.x - 1"。
\item \verb"threadIdx.x"：该变量指定一个线程在一个线程块中的线程指标，其取值范围是从0 到\verb"blockDim.x - 1" 。
\end{itemize}

举一个具体的例子。假如某个核函数的执行配置是\verb"<<<10000, 256>>>"，那么网格大小\verb"gridDim.x"的值为10000，线程块大小\verb"blockDim.x"的值为256。线程块指标\verb"blockIdx.x" 可以取0到9999之间的值，而每一个线程块中的线程指标\verb"threadIdx.x" 可以取0到255之间的值。当\verb"blockIdx.x" 等于0时，所有256 个\verb"threadIdx.x"的值对应第0个线程块；当\verb"blockIdx.x" 等于1时，所有256 个\verb"threadIdx.x"的值对应于第1 个线程块；依此类推。

再次回到Hello World程序。在程序\verb"hello3.cu" 中，我们指派了8个线程，每个线程输出了一行文字，但我们不知道哪一行是由哪个线程输出的。既然每一个线程都有一个唯一的身份标识，那么我们就可以利用该身份标识判断哪一行是由哪个线程输出的。为此，我们将程序改写为Listing \ref{listing:hello4.cu}。

\begin{lstlisting}[language=C++,caption={本章程序hello4.cu中的内容。},label={listing:hello4.cu}]
#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
\end{lstlisting}

编译、运行这个程序，有时输出如下文字：
\begin{verbatim}
    Hello World from block 1 and thread 0.
    Hello World from block 1 and thread 1.
    Hello World from block 1 and thread 2.
    Hello World from block 1 and thread 3.
    Hello World from block 0 and thread 0.
    Hello World from block 0 and thread 1.
    Hello World from block 0 and thread 2.
    Hello World from block 0 and thread 3.
\end{verbatim}
有时输出如下文字：
\begin{verbatim}
    Hello World from block 0 and thread 0.
    Hello World from block 0 and thread 1.
    Hello World from block 0 and thread 2.
    Hello World from block 0 and thread 3.
    Hello World from block 1 and thread 0.
    Hello World from block 1 and thread 1.
    Hello World from block 1 and thread 2.
    Hello World from block 1 and thread 3.
\end{verbatim}
也就是说，有时是第0个线程块先完成计算，有时是第1个线程块先完成计算。这反映了CUDA程序执行时的一个很重要的特征，即每个线程块的计算是相互独立的。不管完成计算的次序如何，每个线程块中的每个线程都进行一次计算。

### Generalization to multi-dimensional grid

细心的读者可能注意到，前面介绍的4个内建变量都用了C++中的结构体（struct）或者类（class）的成员变量的语法。其中：
\begin{itemize}
\item \verb"blockIdx" 和\verb"threadIdx" 是类型为\verb"uint3" 的变量。该类型是一个结构体，具有\verb"x"、\verb"y"、 \verb"z" 这3个成员。所以，\verb"blockIdx.x" 只是3个成员中的一个，另外两个成员分别是
    \verb"blockIdx.y" 和\verb"blockIdx.z"。类似地，\verb"threadIdx.x" 只是3个成员中的一个，另外两个成员分别是
    \verb"threadIdx.y" 和\verb"threadIdx.z"。结构体\verb"uint3"在头文件\verb"vector_types.h"中定义：
    \begin{verbatim}
    struct __device_builtin__ uint3
    {
        unsigned int x, y, z;
    };    
    typedef __device_builtin__ struct uint3 uint3;
    \end{verbatim}
也就是说，该结构体由3个无符号整数类型的成员构成。
\item \verb"gridDim" 和\verb"blockDim" 是类型为\verb"dim3" 的变量。该类型是一个结构体，具有\verb"x"、\verb"y"、 \verb"z" 这3个成员。所以，\verb"gridDim.x" 只是3个成员中的一个，另外两个成员分别是\verb"gridDim.y"和\verb"gridDim.z"。类似地，\verb"blockDim.x"只是3个成员中的一个，另外两个成员分别是\verb"blockDim.y"和\verb"blockDim.z"。结构体\verb"dim3"也在头文件\verb"vector_types.h"定义，除了和结构体\verb"uint3"有同样的3个成员之外，还在使用C++程序的情况下定义了一些成员函数，如下面使用的构造函数。
\end{itemize}
这些内建变量都只在核函数中有效（可见），而且满足如下关系：
\begin{itemize}
\item \verb"blockIdx.x" 的取值范围是从0 到\verb"gridDim.x - 1"。
\item \verb"blockIdx.y" 的取值范围是从0 到\verb"gridDim.y - 1"。
\item \verb"blockIdx.z" 的取值范围是从0 到\verb"gridDim.z - 1"。
\item \verb"threadIdx.x" 的取值范围是从0 到\verb"blockDim.x - 1" 。
\item \verb"threadIdx.y" 的取值范围是从0 到\verb"blockDim.y - 1" 。
\item \verb"threadIdx.z" 的取值范围是从0 到\verb"blockDim.z - 1" 。
\end{itemize}

我们前面介绍过，网格大小和线程块大小是在调用核函数时通过执行配置指定的。在之前的例子中，我们用的执行配置仅仅用了两个整数：
\begin{verbatim}
    <<<grid_size, block_size>>>
\end{verbatim}
我们知道，这两个整数的值将分别赋给内建变量\verb"gridDim.x" 和\verb"blockDim.x"。此时，\verb"gridDim" 和\verb"blockDim" 中没有被指定的成员取默认值1。在这种情况下，网格和线程块实际上都是“一维”的。

也可以用结构体\verb"dim3" 定义“多维”的网格和线程块（这里用了C++中构造函数的语法）：
\begin{verbatim}
    dim3 grid_size(Gx, Gy, Gz);
    dim3 block_size(Bx, By, Bz);
\end{verbatim}
如果第三个维度的大小是1，可以写
\begin{verbatim}
    dim3 grid_size(Gx, Gy);
    dim3 block_size(Bx, By);
\end{verbatim}
例如，如果要定义一个$2 \times 2 \times 1$的网格及$3 \times 2 \times 1$的线程块，可将执行配置中的\verb"grid_size"和\verb"block_size"分别定义为如下结构体变量：
\begin{verbatim}
    dim3 grid_size(2, 2);  // 等价于 dim3 grid_size(2, 2, 1);
    dim3 block_size(3, 2); // 等价于 dim3 block_size(3, 2, 1);
\end{verbatim}
由此产生的核函数中的线程组织见图\ref{figure:threads}。

\begin{figure}[ht]
  \centering
  \captionsetup{font=small}
  \includegraphics[width=\columnwidth]{threads.pdf}\\
  \caption{CUDA核函数中的线程组织示意图。在执行一个核函数时，会产生一个网格，由多个相同大小的线程块构成。该图中展示的是有$2 \times 2 \times 1$个线程块的网格，其中每个线程块包含$3 \times 2 \times 1$个线程。}
  \label{figure:threads}
\end{figure}

多维的网格和线程块本质上还是一维的，就像多维数组本质上也是一维数组一样。
与一个多维线程指标\verb"threadIdx.x"、\verb"threadIdx.y"、\verb"threadIdx.z"对应的一维指标为
\begin{verbatim}
    int tid = threadIdx.z * blockDim.x * blockDim.y + 
              threadIdx.y * blockDim.x + threadIdx.x;
\end{verbatim}
也就是说，\verb"x"维度是最内层的（变化最快），而\verb"z"维度是最外层的（变化最慢）。与一个多维线程块指标\verb"blockIdx.x"、\verb"blockIdx.y"、\verb"blockIdx.z"对应的一维指标没有唯一的定义（主要是因为各个线程块的执行是相互独立的），但也可以类似地定义：
\begin{verbatim}
    int bid = blockIdx.z * gridDim.x * gridDim.y + 
              blockIdx.y * gridDim.x + blockIdx.x;
\end{verbatim}
对于有些问题，如第\ref{chapter:global}章引入的矩阵转置问题，有时使用如下复合线程索引更合适：
\begin{verbatim}
    int nx = blockDim.x * blockIdx.x + threadIdx.x;
    int ny = blockDim.y * blockIdx.y + threadIdx.y;
    int nz = blockDim.z * blockIdx.z + threadIdx.z;
\end{verbatim}

一个线程块中的线程还可以细分为不同的线程束（thread warp）。
一个线程束（即一束线程）是同一个线程块中相邻的\verb"warpSize"个线程。\verb"warpSize"也是一个内建变量，表示线程束大小，其值对于目前所有的GPU架构都是32。所以，一个线程束就是连续的32个线程。具体地说，一个线程块中第0到第31个线程属于第0个线程束，第32到第63个线程属于第1个线程束，依此类推。图\ref{figure:warps}中展示的每个线程块拥有两个线程束。

我们可以通过继续修改Hello World程序来展示使用多维线程块的核函数中的线程组织情况。Listing \ref{listing:hello5.cu}是修改后的代码，在调用核函数时指定了一个$2 \times 4$的两维线程块。程序的输出是：
\begin{verbatim}
    Hello World from block-0 and thread-(0, 0)!
    Hello World from block-0 and thread-(1, 0)!
    Hello World from block-0 and thread-(0, 1)!
    Hello World from block-0 and thread-(1, 1)!
    Hello World from block-0 and thread-(0, 2)!
    Hello World from block-0 and thread-(1, 2)!
    Hello World from block-0 and thread-(0, 3)!
    Hello World from block-0 and thread-(1, 3)!
\end{verbatim}

\begin{figure}[ht]
  \centering
  \captionsetup{font=small}
  \includegraphics[width=\columnwidth]{warp.pdf}\\
  \caption{线程块中相邻的32个线程构成一个线程束。}
  \label{figure:warps}
\end{figure}

\begin{lstlisting}[language=C++,caption={本章程序hello5.cu中的内容。},label={listing:hello5.cu}]
#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}

int main(void)
{
    const dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}
\end{lstlisting}

因为线程块的大小是$2 \times 4$，所以我们知道在核 函数中，\verb"blockDim.x" 的值为2，\verb"blockDim.y" 的值为4。可以看到， \verb"threadIdx.x"的取值范围是从0 到1，而\verb"threadIdx.y" 的取值范围是从0 到3。另外，因为网格大小\verb"gridDim.x"是1，故核函数中\verb"blockIdx.x" 的值只能为0。最后，从输出结果可以确认，\verb"x"维度的线程指标\verb"threadIdx.x"是最内层的（变化最快）。

\subsection{网格与线程块大小的限制}

CUDA中对能够定义的网格大小和线程块大小做了限制。对任何从开普勒到图灵架构的GPU来说，网格大小在x、y和z这3个方向的最大允许值分别为$2^{31}-1$、$65535$和$65535$；线程块大小在x、y和z这3个方向的最大允许值分别为$1024$、$1024$和$64$。另外还要求线程块总的大小，即\verb"blockDim.x"、\verb"blockDim.y"和\verb"blockDim.z"的乘积不能大于$1024$。也就是说，不管如何定义，一个线程块最多只能有$1024$个线程。这些限制是必须牢记的。


## Headers in CUDA

我们知道，在编写C++程序时，往往需要在源文件中包含一些标准的头文件。读者也许注意到了，本章程序包含了C++的头文件\verb"<stdio.h>"，但并没有包含任何CUDA相关的头文件。CUDA中也有一些头文件，但是在使用nvcc编译器驱动编译\verb".cu"文件时，将自动包含必要的CUDA头文件，如\verb"<cuda.h>" 和\verb"<cuda_runtime.h>"。因为\verb"<cuda.h>"包含了\verb"<stdlib.h>"，故用nvcc编译CUDA程序时甚至不需要在\verb".cu"文件中包含\verb"<stdlib.h>"。当然，用户依然可以在\verb".cu"文件中包含\verb"<stdlib.h>"，因为（正确编写的）头文件不会在一个编译单元内被包含多次。本书会从第\ref{chapter:error-check}章开始使用一个用户自定义头文件。

在本书第\ref{chapter:lib}章我们将看到，在使用一些利用CUDA进行加速的应用程序库时，需要包含一些必要的头文件，并有可能还需要指定链接选项。

## Using `nvcc` to compile CUDA programs

CUDA的编译器驱动（compiler driver）nvcc先将全部源代码分离为主机代码和设备代码。主机代码完整地支持C++语法，但设备代码只部分地支持C++。nvcc先将设备代码编译为PTX（Parallel Thread eXecution）伪汇编代码，再将PTX代码编译为二进制的cubin目标代码。在将源代码编译为PTX代码时，需要用选项\verb"-arch=compute_XY"指定一个虚拟架构的计算能力，用以确定代码中能够使用的CUDA功能。在将PTX代码编译为cubin代码时，需要用选项\verb"-code=sm_ZW"指定一个真实架构的计算能力，用以确定可执行文件能够使用的GPU。真实架构的计算能力必须等于或者大于虚拟架构的计算能力。例如，可以用选项
\begin{verbatim}
    -arch=compute_35 -code=sm_60
\end{verbatim}
编译，但不能用选项
\begin{verbatim}
    -arch=compute_60 -code=sm_35
\end{verbatim}
编译（编译器会报错）。如果仅仅针对一个GPU编译程序，一般情况下建议将以上两个计算能力都选为所用GPU的计算能力。

用以上的方式编译出来的可执行文件只能在少数几个GPU中才能运行。选项\verb"-code=sm_ZW"指定了GPU的真实架构为\verb"Z.W"。对应的可执行文件只能在主版本号为\verb"Z"、次版本号大于或等于\verb"W"的GPU中运行。举例来说，由编译选项
\begin{verbatim}
    -arch=compute_35 -code=sm_35
\end{verbatim}
编译出来的可执行文件只能在计算能力为3.5和3.7的GPU中执行，而由编译选项
\begin{verbatim}
    -arch=compute_35 -code=sm_60
\end{verbatim}
编译出来的可执行文件只能在所有帕斯卡架构的GPU中执行。

如果希望编译出来的可执行文件能够在更多的GPU中执行，可以同时指定多组计算能力，每一组用如下形式的编译选项：
\begin{verbatim}
    -gencode arch=compute_XY,code=sm_ZW
\end{verbatim}
例如，用选项
\begin{verbatim}
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_70,code=sm_70
\end{verbatim}
编译出来的可执行文件将包含4个二进制版本，分别对应开普勒架构（不包含比较老的3.0和3.2的计算能力）、麦克斯韦架构、帕斯卡架构和伏特架构。这样的可执行文件称为胖二进制文件（fatbinary）。在不同架构的GPU中运行时会自动选择对应的二进制版本。需要注意的是，上述编译选项假定所使用的CUDA版本支持7.0的计算能力，也就是说至少是CUDA 9.0。如果在编译选项中指定了不被支持的计算能力，编译器会报错。另外需要注意的是，过多地指定计算能力，会增加编译时间和可执行文件的大小。

nvcc有一种称为即时编译（just-in-time compilation）的机制，可以在运行可执行文件时从其中保留的PTX代码临时编译出一个cubin目标代码。要在可执行文件中保留（或者说嵌入）一个这样的PTX代码，就必须用如下方式指定所保留PTX代码的虚拟架构：
\begin{verbatim}
    -gencode arch=compute_XY,code=compute_XY
\end{verbatim}
这里的两个计算能力都是虚拟架构的计算能力，必须完全一致。例如，假如我们处于只有CUDA 8.0的年代（不支持伏特架构），但希望编译出的二进制版本适用于尽可能多的GPU，则可以用如下的编译选项：
\begin{verbatim}
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_60,code=compute_60
\end{verbatim}
其中，前三行的选项分别对应3个真实架构的cubin目标代码，第四行的选项对应保留的PTX代码。这样编译出来的可执行文件可以直接在伏特架构的GPU中运行，只不过不一定能充分利用伏特架构的硬件功能。在伏特架构的GPU中运行时，会根据虚拟架构为6.0的PTX代码即时地编译出一个适用于当前GPU的目标代码。

在学习CUDA编程时，有一个简化的编译选项可以使用：
\begin{verbatim}
    -arch=sm_XY
\end{verbatim}
它等价于
\begin{verbatim}
    -gencode arch=compute_XY,code=sm_XY
    -gencode arch=compute_XY,code=compute_XY
\end{verbatim}
例如，在作者的装有GeForce RTX 2070的计算机中，可以用选项\verb"-arch=sm_75"编译一个CUDA程序。

读者也许注意到了，本章的程序在编译时并没有通过编译选项指定计算能力。这是因为编译器有一个默认的计算能力。以下是各个CUDA版本中的编译器在编译CUDA代码时默认的计算能力：
\begin{itemize}
    \item CUDA 6.0 及更早的：默认的计算能力是1.0。
    \item CUDA 6.5 到CUDA 8.0：默认的计算能力是2.0。
    \item CUDA 9.0 到CUDA 10.2：默认的计算能力是3.0。
\end{itemize}

作者所用的CUDA版本是10.1，故本章的程序在编译时实际上使用了3.0的计算能力。如果用CUDA 6.0进行编译，而且不指定一个计算能力，则会使用默认的1.0的计算能力。此时本章的程序将无法正确地编译，因为从GPU中直接向屏幕打印信息是从计算能力2.0才开始支持的功能。正如在第\ref{chapter:GPU-and-CUDA}章强调过的，本书中的所有示例程序都可以在CUDA9.0-10.2中进行测试。

关于nvcc编译器驱动更多的介绍，请参考如下官方文档：\url{https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc}。
