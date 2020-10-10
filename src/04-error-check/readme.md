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

We cannot use the above method to check errors for CUDA kernels, because there is no return values for CUDA kernels. A method to check CUDA kernels is to add the following two statements after every kernel invocation:

```c++
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
```

The first statement can capture the last error before the second statement, and the second statement can synchronize the host and the device. The reason for using a synchronization between host and device is that kernel launching is asynchronous, which means that the host will continue to execute the statements after launching a CUDA kernel, not waiting for the completion of the kernel execution. The CUDA API function `cudaDeviceSynchronize` forces the host to wait for the completion of the kernel before moving on.

As an example, we check the CUDA kernel in the program [check2kernel.cu](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/04-error-check/check2kernel.cu). When calling the kernel `add` in this program, we intentionally use a block size of 1280, which exceeds the allowed upper bound of 1024. Compile and run this program, I got the following output:

```
        File:       check4kernel.cu
        Line:       36
        Error code: 9
        Error text: invalid configuration argument
```

For this program, if we have not checked the CUDA kernel, we will only see `Has errors`, not knowing the exact reason for not obtaining correct results. Using `cudaDeviceSynchronize` unnecessarily can heavily reduce the performance of a CUDA program. When debugging a CUDA program, we can temporarily set the following environment variable:

```shell
$ export CUDA_LAUNCH_BLOCKING=1        
```

This will make kernel launching synchronous, as if `cudaDeviceSynchronize`  is used after each kernel calling.

## 4.2 Using CUDA-MEMCHECK to check memory errors

CUDA provides a CUDA-MEMCHECK tool set, which can be used in the following way:

```shell
   $ cuda-memcheck --tool memcheck [options] app_name [options] 
   $ cuda-memcheck --tool racecheck [options] app_name [options] 
   $ cuda-memcheck --tool initcheck [options] app_name [options]
   $ cuda-memcheck --tool synccheck [options] app_name [options]
```

Here, `app_name` is the CUDA program we want to debug. For the first, it can be simplified to:

```shell
$ cuda-memcheck [options] app_name [options]
```



As a demonstration, we remove the `if` clause in the program [add3if.cu](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/03-basic-framework/add3if.cu) of last chapter to obtain the [memcheck.cu](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/04-error-check/memcheck.cu) program of this chapter. We compile the program as before and run the executable as follows: 

```shell
$ cuda-memcheck ./a.out
```

The author got a lot of outputs, and the last line reads (the reader might not get the number 36 as below):

```shell
========= ERROR SUMMARY: 36 error
```

This indicates that there are memory errors in the program. If we add the `if` clause back and try again, there will be very simple outputs and the last line should read:

```shell
========= ERROR SUMMARY: 0 errors
```

So you see that CUDA-MEMCHECK can be useful. For more details about CUDA-MEMCHECK, please check the official manual: https://docs.nvidia.com/cuda/cuda-memcheck.

