# Chapter 2: Thread Organization in CUDA

We start with the simplest CUDA program: printing a `Hello World` string from the GPU.

## 2.1 A `Hello World` Program in C++

**To master CUDA C++, one must first master C++**, but we still begin with one of the simplest C++ program: printing a `Hello World` message to the console (screen).

To develop a simple C++ program, one can follow the following steps:
* Write the source code using a text editor (such as `gedit`; you can choose whatever you like).
* Use a compiler to compile the source code to obtain an object file and then use a linker to link the object file and some standard object files to obtain an executable. The compiling and linking processes are usually done with a single command and we will simply call it a compiling process. 
* Run the executable.

Let us first write the following program in a source file named [`hello.cpp`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello.cpp).
```c++
#include <stdio.h>

int main(void)
{
    printf("Hello World!\n");
    return 0;
}
```

In Linux, we can use the following command to compile it:
```shell
$ g++ hello.cpp
```
This will generate an executable named `a.out` in the current directory. One can run this executable by typing
```shell
$ ./a.out
```
Then, one can see the following message printed in the console (screen):
```shell
Hello World!
```
One can also specify a name for the executable, e. g.,
```shell
$ g++ hello.cpp -o hello
```
This will generate an executable named `hello` in the current directory. One can run this executable by typing
```
$ ./hello
```

In Windows with the MSVC compiler `cl.exe`, one can compile the program using the following command in a command prompt:
```shell
$ cl.exe hello.cpp
```
This will generate an executable named `hello.exe`. It can be run using the following command
```shell
$ hello.exe
```

## 2.2 `Hello World` Programs in CUDA

After reviewing the Hello World program in C++, we are ready to discuss similar programs in CUDA.

### 2.2.1 A CUDA program containing host functions only

We actually have already written a valid CUDA program. This is because that the CUDA compiler driver `nvcc` can compile pure C++ code by calling a host compiler (such as `g++` or `cl.exe`). The default suffix for CUDA source files is `.cu` and we thus rename `hello.cpp` as [`hello1.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello1.cu) and use the following command to compile it:
```shell
$ nvcc hello1.cu
```
The output from running the executable is the same as before. We will talk more about `nvcc` in the last section of this chapter. Now the reader only needs to know that `nvcc` can be used to compile CUDA source files with `.cu` suffix.

### 2.2.2 A CUDA program containing a CUDA kernel

Although the file `hello1.cu` was compiled using `nvcc`, that program has not used GPU. We now introduce a program that really used GPU.

We know that GPU is device, which requires a host to give it commands. Therefore, a typical simple CUDA program has the following form:
```c++
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
```c++
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}    
```
The order of the qualifiers, `__global__` and `void`, are not important. That is, we can also write the kernel as:
```c++
void __global__ hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}
```

We then write a main function and call the kernel from the host:
```c++
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
```

The file [`hello2.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello2.cu) can be compiled as follows:
```shell
$ nvcc hello2.cu
```
Run the executable and we will see the following message from the console:
```shell
Hello World from the GPU!
```

We note that the kernel is called in the following way:
```c++
    hello_from_gpu<<<1, 1>>>();
```
There must be an **execution configuration** like `<<<1, 1>>>` between the kernel name and `()`. The execution configuration specifies the number of threads and their organization for the kernel. The threads for a kernel form a **grid**, which can contain multiple **blocks**. Each block in turn can contain multiple threads. The number of blocks within the grid is called the **grid size**. All the blocks in the grid have the same number of threads and this number is called the **block size**. Therefore, the total number of threads in a grid is the product of the grid size and the block size. For a simple execution configuration `<<<grid_size, block_size>>>` with two integer numbers `grid_size` and `block_size` (we will see more general execution configurations soon), the first number `grid_size` is the grid size and the second number `block_size` is the block size. In our `hello2.cu` program, the execution configuration `<<<1, 1>>>` means that both the grid size and the block size are 1 and there is only `1 * 1 = 1` thread used for the kernel.

* The `printf()` function from the C++ library `<stdio.h>` (can also be written as `<cstdio>`) can be directly used in kernels. However, one cannot use functions from the `<iostream>` library in kernels. The statement
```c++
    cudaDeviceSynchronize();
```
after the kernel call is used to **synchronize the host and the device**, making sure that the output stream for the `printf` function has been flushed before returning from the kernel to the host. Without a synchronization like this, the host will not wait for the completion of kernel execution and the message would not be output to console. `cudaDeviceSynchronize()` is one of the many CUDA runtime API functions we will learn during the course of this book. The need for synchronization here reflects the **asynchronous nature of kernel launching**, but we will not bother to elaborate on it until Chapter 11.

## 2.3 Thread organization in CUDA 

### 2.3.1 A CUDA kernel using multiple threads

There are many cores in a GPU and one can assign many threads for a kernel, if needed. The following program [`hello3.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello3.cu) used a grid with 2 blocks for the kernel, and each block has 4 threads:
```c++
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
```c++
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
* `blockIdx` and `threadIdx` are of type `uint3`, which is defined in `vector_types.h` as:
```c++
    struct __device_builtin__ uint3
    {
        unsigned int x, y, z;
    };    
    typedef __device_builtin__ struct uint3 uint3;
```
Therefore, apart from `blockIdx.x`, we also have `blockIdx.y` and `blockIdx.z`. Similarly, apart from `threadIdx.x`, we also have `threadIdx.y` and `threadIdx.z`.
* `gridDim` and `blockDim` are of type `dim3`, which is similar to `uint3` and has some constructors which will be introduced soon. Therefore, apart from `gridDim.x`, we also have `gridDim.y` and `gridDim.z`. Similarly, apart from `blockDim.x`, we also have `blockDim.y` and `blockDim.z`.
* These built-in variables thus represent indices or sizes in three dimensions: `x`, `y`, and `z`. **All these built-in variables are only visible within CUDA kernels.**

We can use the constructors of the struct `dim3` to define multi-dimensional grids and blocks:
```c++
    dim3 grid_size(Gx, Gy, Gz);
    dim3 block_size(Bx, By, Bz);
```
If the size of the `z` dimension is 1, we can simplify the above definitions to:
```c++
    dim3 grid_size(Gx, Gy);
    dim3 block_size(Bx, By);
```

To demonstrate the usage of a multi-dimensional block, we write our last version of the Hello World program [`hello5.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello5.cu):
```c++
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
```

The output of this program is:
```
    Hello World from block-0 and thread-(0, 0)!
    Hello World from block-0 and thread-(1, 0)!
    Hello World from block-0 and thread-(0, 1)!
    Hello World from block-0 and thread-(1, 1)!
    Hello World from block-0 and thread-(0, 2)!
    Hello World from block-0 and thread-(1, 2)!
    Hello World from block-0 and thread-(0, 3)!
    Hello World from block-0 and thread-(1, 3)!
```

The reader may notice a well defined order for `threadIdx.x` and `threadIdx.y` here. If we label the lines as 0-7 from top to down, this label can be calculated as `threadIdx.y * blockDim.x + threadIdx.x = threadIdx.y * 2 + threadIdx.x`:
```
    Hello World from block-0 and thread-(0, 0)! // 0 = 0 * 2 + 0
    Hello World from block-0 and thread-(1, 0)! // 1 = 0 * 2 + 1
    Hello World from block-0 and thread-(0, 1)! // 2 = 1 * 2 + 0
    Hello World from block-0 and thread-(1, 1)! // 3 = 1 * 2 + 1
    Hello World from block-0 and thread-(0, 2)! // 4 = 2 * 2 + 0
    Hello World from block-0 and thread-(1, 2)! // 5 = 2 * 2 + 1
    Hello World from block-0 and thread-(0, 3)! // 6 = 3 * 2 + 0
    Hello World from block-0 and thread-(1, 3)! // 7 = 3 * 2 + 1
```

In general, the one-dimensional index `tid` of a thread is related to the multi-dimensional indices of the thread via the the following relation:
```c++
    int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
```
**This is an important indexing rule**, which will be relevant to our discussion of **coalesced memory access** in Chapter 7.

### 2.3.4 Limits on the grid and block sizes

For all the GPUs starting from the Kepler architecture, the grid size is limited to 
```c++
  gridDim.x <= 2^{31}-1
  gridDim.y <= 2^{16}-1 = 65535
  gridDim.z <= 2^{16}-1 = 65535
```
and the block size is limited to
```c++
  blockDim.x <= 1024
  blockDim.y <= 1024
  blockDim.z <= 64
```
Besides this, there is an important limit on the following product:
```c++
  blockDim.x * blockDim.y * blockDim.z <= 1024
```

**It is important to remember the above limits.**

## 2.5 Using `nvcc` to compile CUDA programs

### 2.5.1 Headers in CUDA

When using `nvcc` to compile a `.cu` source file, some CUDA related headers, such as `<cuda.h>` and `<cuda_runtime.h>` will be automatically included. In this book, we deal with pure CUDA code: all the source files will be `.cu` files and we only use `nvcc` to compile them. We therefore do not bother to figure out what headers are needed. I might extend this book by considering using mixed `.cu` and `.cpp` files for a program in a future version. 

### 2.5.2 Some important flags for `nvcc` 

The CUDA compiler driver `nvcc` first separates the source code into host code and device code. The host code will be compiled by a host C++ compiler such as `cl.exe` or `g++`. `nvcc` will first compile the device code into an intermediate PTX（Parallel Thread eXecution）code, and then compile the PTX code into a **cubin** binary. 

When compiling a device code into a PTX code, a flag `-arch=compute_XY` to `nvcc` is needed to specify the compute capability of a **virtual architecture**, which determines the CUDA features that can be used. When compiling a PTX code into a cubin binary, a flag `-code=sm_ZW` is needed to specify the compute capability of a **real architecture**, which determines the GPUs on which the binary can run. **The compute capability of the real architecture must be no less than that of the virtual architecture.** For example, 
```
$ nvcc -arch=compute_60 -code=sm_70 xxx.cu
```
is ok, but 
```
$ nvcc -arch=compute_70 -code=sm_60 xxx.cu
```
will result in errors. Usually, the two compute capabilities are set to the same, e.g.,
```
$ nvcc -arch=compute_70 -code=sm_70 xxx.cu
```

An executable compiled in the above way can only be run in GPUs with compute capability `Z.V`, where `V >= W`. For example, an executable compiled using 
```
$ nvcc -arch=compute_60 -code=sm_60 xxx.cu
```
can only be run in GPUs with Pascal architectures.

If one hopes that an executable can be run in more GPUs, one can specify more flags in the following way:
```
$ nvcc -gencode arch=compute_60,code=sm_60 \
       -gencode arch=compute_70,code=sm_70 \
       -gencode arch=compute_80,code=sm_80 \
       xxx.cu
```

There is a mechanism called just-in-time compilation in `nvcc`, which can get a cubin binary from a PTX code *just in time*, if such a PTX code has been kept. To keep such a PTX code, one must use the following flag:
```
    -gencode arch=compute_XY,code=compute_XY
```
For example, an executable compiled using 
```
$ nvcc -gencode arch=compute_60,code=sm_60 \      # generate a cubin binary for Pascal arthitecture
       -gencode arch=compute_60,code=compute_60 \ # generate a PTX that can be just-in-time compiled to cubin binary for newer architectures
       xxx.cu
```
can be run in GPUs with any architecture no less than Pascal. There is a simplified version for the above command:
```
$ nvcc -arch=sm_60 xxx.cu
```
We will use this simplified version all over this book.

The reader might have noticed that we have not used any flag when compiling our Hello World programs using `nvcc`. This is because each CUDA version has a default flag for the compute capability:
* CUDA 6.0 and older: default to compute capability 1.0
* CUDA 6.5-8.0: default to compute capability 2.0
* CUDA 9.0-10.2: default to compute capability 3.0
* CUDA 11.0: default to compute capability 3.5

The author used CUDA 10.1, which defaults to compute capability 3.0. However, we will specify a compute capability for our future programs in this book.

For more details about `nvcc`, see the official manual: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc.
