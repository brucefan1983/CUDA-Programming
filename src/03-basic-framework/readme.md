**unfinished...**

# Chapter 3 Basic framework of simple CUDA programs

## 3.1 An example: adding up two arrays

We consider a simple task: adding up two arrays of the same length (same number of elements). We first write a C++ program [add.cpp](https://github.com/brucefan1983/CUDA-Programming/tree/master/src/03-basic-framework/add.cpp) solving this problem. It can be compiled by using `g++` (or `cl.exe`):

```
g++ add.cpp
```

Running the executable, we will see the following message on the screen:
```
No errors
```
which indicates that the calculations have been done in an expected way. The reader should be able to understand this program without difficulty, otherwise he/she needs to gain sufficient knowledge of C++ programming first.

## 3.2 Basic framework of simple CUDA programs

For a simple CUDA program written in a single source file, the basic framework is as follows:
```c++
header inclusion
const or macro definition
declarations of C++ functions and CUDA kernels

int main()
{
    allocate host and device memory
    initialize data in host memory
    transfer data from host to device
    launch (call) kernel to do calculations in the device
    transfer data from device to host
    free host and device memory
}

definitions of C++ functions and CUDA kernels
```

We first write a CUDA program [add1.cu](https://github.com/brucefan1983/CUDA-Programming/tree/master/src/03-basic-framework/add.cu) which does the same calculations as the C++ program [add.cpp](https://github.com/brucefan1983/CUDA-Programming/tree/master/src/03-basic-framework/add.cpp). This CUDA program can be compiled as follows:

```shell
$ nvcc -arch=sm_75 add1.cu 
```
Executing the executable will produce the same output as the C++ program:
```
No errors
```

We will describe the CUDA program [add1.cu](https://github.com/brucefan1983/CUDA-Programming/tree/master/src/03-basic-framework/add.cu) in detail in the following sections.

### 3.2.1 Memory allocation in device

In our CUDA program, we defined three pointers 

```c++
double *d_x, *d_y, *d_z;
```

and used the `cudaMalloc()` function to allocate memory in device. This is a CUDA runtime API function. Every CUDA runtime API function begins with `cuda`. Here is the online manual for all the CUDA runtime functions: https://docs.nvidia.com/cuda/cuda-runtime-api.

The prototype of `cudaMalloc()` is:

```c++
cudaError_t cudaMalloc(void **address, size_t size);
```

Here, `address` is the address of the pointer (so it is a double pointer), `size` is the number of bytes to be allocated, and `cudaSuccess` is a return value indicating whether there is error when calling this function. We will ignore this return value in this Chapter and discuss it in the next Chapter. In the CUDA program, we have used this function to allocate memory for the three pointers:

```c++
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);
```

Here, `M` is `sizeof(double) * N`, where `N` is the number of elements in an array, and `sizeof(double)` is the  memory size (number of bytes) for a double-precision floating point number. The type conversion `(void **)` can be omitted, i.e., we can change the above lines to:

```c++
    cudaMalloc(&d_x, M);
    cudaMalloc(&d_y, M);
    cudaMalloc(&d_z, M);
```

The reason for using a pointer to pointer for the first parameter in this function is that we need to change the value of the pointer itself, other than the value in the memory pointed by the pointer.

Memory allocated by `cudaMalloc()` needs to be freed by using the `cudaFree()` function:

```c++
cudaError_t cudaFree(void* address);   
```

Note that the argument here is a pointer, not a double pointer.

### 3.2.2 Data transfer between host and device

We can transfer (copy) some data from host to device after allocating the device memory, see lines 29-30 in  [add1.cu](https://github.com/brucefan1983/CUDA-Programming/tree/master/src/03-basic-framework/add.cu). Here we used the CUDA runtime API function `cudaMemcpy()` with the following prototype:

```c++
cudaError_t cudaMemcpy( 	
    void                *dst,
    const void          *src,
    size_t              count,
    enum cudaMemcpyKind kind);
```

Here, `dst` is the address of the destination (to be transferred to), `src` is the address of the source (to be transferred from), `count` is the number of bytes to transferred , and `kind` indicates the direction of the data transfer. The possible values of the enum parameter `kind` include `cudaMemcpyHostToHost`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`, and `cudaMemcpyDefault`. The meanings of the first 4 are obvious and for the last one, it means that transfer direction will be automatically inferred from the pointers `dst` and `src`. This automatic process requires that the host system is 64 bit supporting unified virtual addressing. Therefore, one can also use `cudaMemcpyDefault` in lines 29-30. 

After calling the kernel at line 34, we use the `cudaMemcpy` function to transfer some data from device to host, where the last parameter should be `cudaMemcpyDeviceToHost` or `cudaMemcpyDefault`. 

In the [add2wrong.cu](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/03-basic-framework/add2wrong.cu) program, the author intentionally changed `cudaMemcpyHostToDevice` to `cudaMemcpyDeviceToHost`. The reader can try to compile and run it to see what happens.

### 3.2.3 Correspondence between data and threads in CUDA kernel

Lines 32-34 defined the execution configuration for the kernel: a block size of 128 and a grid size of 10^8/128. 

Now we check the `add` functions in  `add.cpp` and `add1.cu`. We see that there is a `for` loop in the host function, but not in the kernel. In the host `add` function, we need to loop over each element of the arrays, and thus need a `for` loop. In the device `add` function (the kernel), this `for`-loop is gone. This is because we have many threads in the kernel and each thread will do the same calculation, but with different data, int the so-called "single-instruction-multiple-threads" way. In the kernel, we define a one-dimensional index `n` in the following way:

```c++
const int n = blockDim.x * blockIdx.x + threadIdx.x;
```

This provides a correspondence between the data index `n` and the thread indices `blockIdx` and `threadIdx`. With this `n` defined, we can simply use it to access the data stored in the arrays:

```c++
z[n] = x[n] + y[n];
```

We stress again that even if each thread executes this same statement, the value of `n` is different for different threads. 

### 3.2.3 Some requirements for kernels

Kernels are the most important aspect in CUDA programming. Here we list a few general requirements for CUDA kernels:

* A kernel must return `void`.
* A kernel must be decorated by `__global__`.
* Function name for a kernel can be overloaded.
* The number of parameters for a kernel must be fixed. 
* We can pass normal values to a kernel, which is visible for each thread. We will know that these parameters will be read through the constant cache in Chapter 6.
* Pointers passed to a kernel must point to device memory, unless unified memory is used (to be discussed in Chapter 12).
* Kernels cannot be class member functions. Usually, one wraps kernels within class members.
* A kernel cannot call another kernel, unless dynamic parallelism is used, but we will not touch this topic in this book.
* One must provide an execution configuration when launching a kernel.



**up to here**

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
