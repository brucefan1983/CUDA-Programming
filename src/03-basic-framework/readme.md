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

### 3.2.4 Some requirements for kernels

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

### 3.2.5 The necessity of `if` statements in most kernels

The kernel in `add1.cu` does not use the parameter `N`. When `N` can be divided by `blockDim.x`, this is OK. Otherwise, we will be in trouble. To show this, we change the value of `N` from 10^8 to 10^8+1. If we want to have enough threads for our task and still use one thread for one element in an array, the grid size should be 10^8/128 + 1 = 781250 + 1 = 781251. In general, when the number of elements cannot be divided by the block size, the grid size can be calculated in one of the following ways:

```c++
int grid_size  = (N - 1) / block_size + 1;
int grid_size  = (N + block_size - 1) / block_size;
```

   They are both equivalent to the following statement:

```c++
int grid_size = (N % block_size == 0) 
              ? (N / block_size) 
              : (N / block_size + 1);
```

Because now the number of threads (10^8+128) exceeds the number of elements (10^8+1), we must use an `if` statement to avoid manipulating invalid addresses:

```c++
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
```

It can be equivalently written as:

```c++
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= N) return;
    z[n] = x[n] + y[n];
```

See the program [add3if.cu](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/03-basic-framework/add3if.cu) for the whole code. 

## 3.3 User-defined device functions

Kernels can call functions without a execution configuration, which are called device functions. These functions are called within kernels and executed in devices. To distinguish the various functions in a CUDA program, some execution space specifiers are introduced:

* Functions decorated as `__global__` are kernels, which are called from host and executed in device. 
* Functions decorated as `__device__` are device functions, which are called from kernels and executed in device. 
* Functions decorated as `__host__` are device functions, which are called from kernels and executed in device. This is usually used together with `__host__` to indicate that a function is simultaneously a host function and a device function. Compilers will generate both versions.
* It is apparent that `__device__` cannot be used together with `__global`.
* It is apparent that `__host__` cannot be used together with `__global`.
* `__noinline__` and `__forceinline` can be used for a device function to suggest the compiler treat it as a non-inline or inline function.

The program [add4device.cu](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/03-basic-framework/add4device.cu) demonstrates the definition and use of device functions, using different styles of returning values. 


