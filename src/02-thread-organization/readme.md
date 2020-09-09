**Unfinished...**

# Chapter 2: Thread Organization in CUDA

We start with the simplest CUDA program: printing a `Hello World` string from the GPU.

## 2.1 A `Hello World` Program in C++

**To master CUDA C++, one must first master C++**, but we still begin with one of the simplest C++ program: printing a `Hello World` message to the console (screen).

To develop a simple C++ program, one can follow the following steps:
* Write the source code using a text editor (such as `gedit`; you can choose whatever you like).
* Use a compiler to compile the source code to obtain an object file and then use a linker to link the object file and some standard object files to obtain an executable. The compiling and linking processes are usually done with a single command and we will simply call it a compiling process. 
* Run the executable.

Let us first write the following program in a source file named [`hello.cpp`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello.cpp).
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
$ hello.exe
```

## 2.2 `Hello World` Programs in CUDA

After reviewing the Hello World program in C++, we are ready to discuss similar programs in CUDA.

### 2.2.1 A CUDA program containing host functions only

We actually have already written a valid CUDA program. This is because that the CUDA compiler driver `nvcc` can compile pure C++ code by calling a host compiler (such as `g++` or `cl.exe`). The default suffix for CUDA source files is `.cu` and we thus rename `hello.cpp` as [`hello1.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello1.cu) and use the following command to compile it:
```
$ nvcc hello1.cu
```
The output from running the executable is the same as before. We will talk more about `nvcc` in the last section of this chapter. Now the reader only needs to know that `nvcc` can be used to compile CUDA source files with `.cu` suffix.

### 2.2.2 A CUDA program containing a CUDA kernel

Although the file `hello1.cu` was compiled using `nvcc`, that program has not used GPU. We now introduce a program that really used GPU.

We know that GPU is device, which requires a host to give it commands. Therefore, a typical simple CUDA program has the following form:
```
int main(void)
{
    host code
　  calling CUDA kernel(s)
　  host code
    return 0;
}
```

A `CUDA kernel` (or simply `kernel`) is a function that is called by the host and executes in the device. There are many rules for defining a kernel, but now we only need to know that it must be decorated by the qualifiers `__global__` and `void`. Here, `__global__` indicates the function is a `kernel` and `void` means that **a CUDA kernel cannot return values**. Inside a kernel, **nearly** all C++ constructs are allowed.

Following the above requirements, we write a kernel that prints a message to the console:
```
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}    
```
The order of the qualifiers, `__global__` and `void`, are not important. That is, we can also write the kernel as:
```
void __global__ hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}
```

We then write a main function and call the kernel from the host:
```
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
```
 
The file [`hello2.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello2.cu) can be compiled as follows:
```
$ nvcc hello2.cu
```
Run the executable and we will see the following message from the console:
```
Hello World from the GPU!
```

We note that the kernel is called in the following way:
```
    hello_from_gpu<<<1, 1>>>();
```
There must be an **execution configuration** like `<<<1, 1>>>` between the kernel name and `()`. The execution configuration specifies the number of threads and their organization for the kernel. The threads for a kernel form a **grid**, which can contain multiple **blocks**. Each block in turn can contain multiple threads. The number of blocks within the grid is called the **grid size**. All the blocks in the grid have the same number of threads and this number is called the **block size**. Therefore, the total number of threads in a grid is the product of the grid size and the block size. For a simple execution configuration `<<<grid_size, block_size>>>` with two integer numbers `grid_size` and `block_size` (we will see more general execution configurations soon), the first number `grid_size` is the grid size and the second number `block_size` is the block size. In our `hello2.cu` program, the execution configuration `<<<1, 1>>>` means that both the grid size and the block size are 1 and there is only `1 * 1 = 1` thread used for the kernel.

* The `printf()` function from the C++ library `<stdio.h>` (can also be written as `<cstdio>`) can be directly used in kernels. However, one cannot use functions from the `<iostream>` library in kernels. The statement
```
    cudaDeviceSynchronize();
```
after the kernel call is used to **synchronize the host and the device**, making sure that the output stream for the `printf` function has been flushed before returning from the kernel to the host. Without a synchronization like this, the host will not wait for the completion of kernel execution and the message would not be output to console. `cudaDeviceSynchronize()` is one of the many CUDA runtime API functions we will learn during the course of this book. The need for synchronization here reflects the **asynchronous nature of kernel launching**, but we will not bother to elaborate on it until Chapter 11.

## 2.3 Thread organization in CUDA 

### 2.3.1 A CUDA kernel using multiple threads

There are many cores in a GPU and one can assign many threads for a kernel, if needed. The following program [`hello3.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello3.cu) used a grid with 2 blocks for the kernel, and each block has 4 threads:
```
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
```

The total number of threads used in the kernel is thus `2 * 4 = 8`. The code in the kernel is executed in a way called "single-instruction-multiple-threads", which means every thread in the kernel (or in the grid) executes the same sequence of instructions (we will talk more about this in Chapter 10). Therefore, running the executable of this program would print the following text to the console:
```
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
```

Every line above corresponds to one thread. But the reader may ask: which line was produced by which thread? We will answer this question below.

### 2.3.2 Using thread indices in a CUDA kernel

Every thread in a kernel has a unique identity, or index. Because we have used two numbers (grid size and block size) in the execution configuration, every thread in the kernel should also be identified by two numbers. In the kernel, the grid size and block size are stored in the built-in variables `gridDim.x` and `blockDim.x`, respectively. A thread can be identified by the following built-in variables:
* `blockIdx.x`: this variable specify the **block index of the thread within a grid**, which can take values from 0 to `gridDim.x - 1`.
* `threadIdx.x`: this variable specify the **thread index of the thread within a block**, which can take values from 0 to `blockDim.x - 1`.

Consider a kernel called with an execution configuration of `<<<10000, 256>>>`, we then know that the grid size `gridDim.x` is 10000，and the block size `blockDim.x` is 256. The block index `blockIdx.x` of a thread in the kernel can thus take values from 0 to 9999, and the thread index `threadIdx.x` of a thread can take values from 0 to 255. 

Returning to our `hello3.cu` program, we have assigned 8 threads to the kernel and each thread printed one line of text, but we didn't know which line was from which thread. Now that we know every thread in the kernel can be uniquely identified, we could use this to tell us which line was from which thread. To this end, we rewrite the program to get a new one, as in [`hello4.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello4.cu):
```
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
```

Running the executable for this program, sometimes we get the following output,
```
    Hello World from block 1 and thread 0.
    Hello World from block 1 and thread 1.
    Hello World from block 1 and thread 2.
    Hello World from block 1 and thread 3.
    Hello World from block 0 and thread 0.
    Hello World from block 0 and thread 1.
    Hello World from block 0 and thread 2.
    Hello World from block 0 and thread 3.
```
and sometimes we get the following output,
```
    Hello World from block 0 and thread 0.
    Hello World from block 0 and thread 1.
    Hello World from block 0 and thread 2.
    Hello World from block 0 and thread 3.
    Hello World from block 1 and thread 0.
    Hello World from block 1 and thread 1.
    Hello World from block 1 and thread 2.
    Hello World from block 1 and thread 3.
```
That is, sometimes block 0 finishes the instructions first, and sometimes block 1 finishes the instructions first. This reflects a very important feature of the execution of CUDA kernels, i.e., **every block in the grid is independent of each other**. 

### 2.3.3 Generalization to multi-dimensional grids and blocks

The reader may have noticed that the 4 built-in variables introduced above used the `struct` or `class` syntax in C++. This is true:
细心的读者可能注意到，前面介绍的4个内建变量都用了C++中的结构体（struct）或者类（class）的成员变量的语法。其中：
* `blockIdx` and `threadIdx` are of type `uint3`, which is defined in `vector_types.h` as:
```
    struct __device_builtin__ uint3
    {
        unsigned int x, y, z;
    };    
    typedef __device_builtin__ struct uint3 uint3;
 ```
Therefore, apart from `blockIdx.x`, we also have `blockIdx.y` and `blockIdx.z`. Similarly, apart from `threadIdx.x`, we also have `threadIdx.y` and `threadIdx.z`.
* `gridDim` and `blockDim` are of type `dim3`, which is similar to `uint3` and has some constructors which will be introduced soon. Therefore, apart from `gridDim.x`, we also have `gridDim.y` and `gridDim.z`. Similarly, apart from `blockDim.x`, we also have `blockDim.y` and `blockDim.z`.

**All these built-in variables are only visibal within CUDA kernels.**

 **I am up to here...**
 
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


## 2.4 Headers in CUDA

我们知道，在编写C++程序时，往往需要在源文件中包含一些标准的头文件。读者也许注意到了，本章程序包含了C++的头文件\verb"<stdio.h>"，但并没有包含任何CUDA相关的头文件。CUDA中也有一些头文件，但是在使用nvcc编译器驱动编译\verb".cu"文件时，将自动包含必要的CUDA头文件，如\verb"<cuda.h>" 和\verb"<cuda_runtime.h>"。因为\verb"<cuda.h>"包含了\verb"<stdlib.h>"，故用nvcc编译CUDA程序时甚至不需要在\verb".cu"文件中包含\verb"<stdlib.h>"。当然，用户依然可以在\verb".cu"文件中包含\verb"<stdlib.h>"，因为（正确编写的）头文件不会在一个编译单元内被包含多次。本书会从第\ref{chapter:error-check}章开始使用一个用户自定义头文件。

在本书第\ref{chapter:lib}章我们将看到，在使用一些利用CUDA进行加速的应用程序库时，需要包含一些必要的头文件，并有可能还需要指定链接选项。

## 2.5 Using `nvcc` to compile CUDA programs

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
