# `CUDA`-Programming
Source codes for my `CUDA` programming book

## 1. Warning
* The codes are still under development.

## 2. About the book:
  * To be published by in 2020.
  * The language for the book is **Chinese**.
  * Covers **from Kepler to Turing (compute capability 3.0-7.5)**.
  * Tested in both **Windows** and **Linux**.
  * I assume that the readers
    * are familiar with `C++` (for the whole book)
    * have studied **physics** and **mathematics** at the undergraduate level (for some chapters)
    
## 3. My testing systems
* Linux: the host compiler we used is `g++`.
* Windows: We only use the command prompt `CMD` in Windows and the host compiler is `cl` from Visual Studio. When using `nvcc` to compile a CUDA code, you may (or may not) need to add the compiling flag `-Xcompiler "/wd 4819"` to suppress warnings related to unicode.
* I installed `CUDA` 10.1 update2.


## 4. Table of contents and list of source codes:

### Chapter 1: Introduction to GPU hardware and `CUDA` programming tools
There is no source code for this chapter.


### Chapter 2: Thread organization in `CUDA`
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `hello.cpp` | writing a Hello Word program in `C++` |`g++ hello.cpp` in Linux or `cl hello.cpp` in Windows |`./a.out` in Linux or `hello` in Windows |
| `hello1.cu` | a valid `C++` program is also a valid `CUDA` program | `nvcc hello1.cu` |`./a.out` in Linux or `a` in Windows |
| `hello2.cu` | write a simple `CUDA` kernel and call `printf()` within it | `nvcc hello2.cu` |`./a.out` in Linux or `a` in Windows |
| `hello3.cu` | using multiple threads in a block | `nvcc hello3.cu` |`./a.out` in Linux or `a` in Windows |
| `hello4.cu` | using multiple blocks in a grid | `nvcc hello4.cu` |`./a.out` in Linux or `a` in Windows |
| `hello5.cu` | using a 2D block | `nvcc hello5.cu` |`./a.out` in Linux or `a` in Windows |


### Chapter 3: The basic framework of a `CUDA` program
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `add.cpp`      | adding up two arrays using `C++` |`g++ add.cpp` in Linux or `cl add.cpp` in Windows |`./a.out` in Linux or `add` in Windows |
| `add1.cu`      | adding up two arrays using `CUDA` | `nvcc add1.cu` in Linux or `nvcc -Xcompiler "/wd 4819" add1.cu` in Windows |`./a.out` in Linux or `a` in Windows |
| `add2wrong.cu` | what if the memory transfer direction is wrong? | `nvcc add2wrong` in Linux or `nvcc -Xcompiler "/wd 4819" add2wrong.cu` in Windows |`./a.out` in Linux or `a` in Windows |
| `add3if.cu`    | when do we need an if statement in the kernel? | `nvcc add3if.cu` in Linux or `nvcc -Xcompiler "/wd 4819" add3if.cu` in Windows |`./a.out` in Linux or `a` in Windows |
| `add4device.cu`| how to define and call `__device__` functions? | `nvcc add4device.cu` in Linux or `nvcc -Xcompiler "/wd 4819" add4device.cu` in Windows |`./a.out` in Linux or `a` in Windows |


### Chapter 4: Error checking
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `check1api.cu`    | how to check `CUDA` runtime API calls? | `nvcc check1api.cu` in Linux or `nvcc -Xcompiler "/wd 4819" check1api.cu` in Windows |`./a.out` in Linux or `a` in Windows |
| `check2kernel.cu`    | how to check `CUDA` runtime API calls? | `nvcc check2kernel.cu` in Linux or `nvcc -Xcompiler "/wd 4819" check2kernel.cu` in Windows |`./a.out` in Linux or `a` in Windows |
| `memcheck.cu`    | how to check `CUDA` runtime API calls? | `nvcc memcheck.cu` in Linux or `nvcc -Xcompiler "/wd 4819" memcheck.cu` in Windows |`cuda-memcheck ./a.out` in Linux or `cuda-memcheck a` in Windows |


### Chapter 5: The crucial ingredients for obtaining speedup
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `add1cpu.cu`              | timing `C++` code | `nvcc -O3 [-DUSE_DP] add1cpu.cu` in Linux or `nvcc -O3 [-DUSE_DP] -Xcompiler "/wd 4819" add1cpu.cu` in Windows |`./a.out` in Linux or `a` in Windows |
| `add2gpu.cu`               | timing `CUDA` kernel using Events | `nvcc -O3 [-DUSE_DP] add2gpu.cu` in Linux or `nvcc -O3 [-DUSE_DP] -Xcompiler "/wd 4819" add2gpu.cu` in Windows |`./a.out` in Linux or `a` in Windows |
| `add3memcpy.cu`               | timing `CUDA` code using Events and `nvprof` | `nvcc -O3 [-DUSE_DP] add3memcpy.cu` in Linux or `nvcc -O3 [-DUSE_DP] -Xcompiler "/wd 4819" add3memcpy.cu` in Windows |`[nvprof] ./a.out` in Linux or `[nvprof] a` in Windows|
| `arithmetic1cpu.cu`       | increasing arithmetic intensity in `C++` | `nvcc -O3 [-DUSE_DP] arithmetic1cpu.cu` in Linux or `nvcc -O3 [-DUSE_DP] -Xcompiler "/wd 4819" arithmetic1cpu.cu` in Windows |`./a.out` in Linux or `a` in Windows |
| `arithmetic2gpu.cu`        | increasing arithmetic intensity in `CUDA` | `nvcc -O3 [-DUSE_DP] arithmetic2gpu.cu` in Linux or `nvcc -O3 [-DUSE_DP] -Xcompiler "/wd 4819" arithmetic2gpu.cu` in Windows |`./a.out` in Linux or `a` in Windows |


### Chapter 6: Memory organization in `CUDA`
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `global.cu`           | how to use static global memory? | `nvcc global.cu` in Linux or `nvcc -Xcompiler "/wd 4819" global.cu` in Windows |`./a.out` in Linux or `a` in Windows |
| `device_query.cu`     | how to query some properties of your GPU? | `nvcc device_query.cu` in Linux or `nvcc -Xcompiler "/wd 4819" device_query.cu` in Windows |`./a.out` in Linux or `a` in Windows |



### Chapter 7: Using global memory: matrix transpose
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `matrix.cu`           | effects of coalesced and uncoalesced memory accesses |

### Chapter 8: Using shared memory: reduction and matrix transpose
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `reduce_cpu.cu`              | reduction in `C++` |
| `reduce_gpu.cu`              | doing reduction in CUDA |
| `bank_conflict.cu`           | how to avoid shared memory bank conflict |


### Chapter 9: Using atomic functions
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `reduce.cu`        | using `atomicAdd` for reduction (good or bad?) |
| `neighbor_cpu.cu`  | neighbor list construction using CPU |
| `neighbor_gpu.cu`  | neighbor list construction using GPU, with and without using `atomicAdd` |


### Chapter 10: Using warp-level functions
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `reduce7syncwarp.cu` | using the `__syncwarp()` function instead of the `__syncthreads()` function within warps |
| `reduce8shfl.cu`     | using the `__shfl_down_sync()` or the `__shfl_xor_sync()` function for warp reduction |
| `reduce9cp.cu`       | using the cooperative groups |


### Chapter 11: `CUDA` streams 
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `host_kernel.cu`     | overlapping host and device computations |
| `kernel_kernel.cu`   | overlaping multiple kernels |
| `kernel_transfer.cu` | overlaping kernel execution and memory transfer |


### Chapter 12: Using `CUDA` libraries
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `thrust_scan_vector.cu`  | using the device vector in `thrust` |
| `thrust_scan_pointer.cu` | using the device pointer in `thrust` |
| `cublas_gemm.cu`         | matrix multiplication in `cuBLAS` |
| `cusolver.cu`            | matrix eigenvalues in `cuSolver` |
| `curand_host1.cu`        | uniform random numbers in `cuRAND` |
| `curand_host2.cu`        | Gaussian random numbers in `cuRAND` |


### Chapter 13: Unified memory programming 
| file        | what to learn? | how to compile? | how to run? |
|:------------|:---------------|:---------------|:---------------|
| `add_unified.cu` | using unified memory |


### Chapter 14: Summary for developing and optimizing `CUDA` programs 
There is no source code for this chapter.


### Chapter 15: Introduction to molecular dynamics simulation
There is no source code for this chapter.


### Chapter 16: A `C++` program for molecular dynamics simulation
How to compile and run?
  * type `make` to compile
  * type `./ljmd 8 10000` to run
  * type `plot_results` in Matlab command window to check the results


### Chapter 17: MD code: only accelerating the force-evaluation part
How to compile and run?
  * type `make` to compile
  * type `./ljmd 40 10000` to run
  * type `plot_results` in Matlab command window to check the results


### Chapter 18: MD code: accelerating the whole code
How to compile and run?
  * type `make` to compile
  * type `./ljmd 40 10000` to run
  * type `plot_results` in Matlab command window to check the results


### Chapter 19: MD code: various optimizations
How to compile and run?
  * type `make` or `make -f makefile.ldg` or `make -f makefile.fast_math` to compile
  * type `./ljmd 40 10000` to run
  * type `plot_results` in Matlab command window to check the results


### Chapter 20: MD code: using unified memory
How to compile and run?
  * type `make` or `make -f makefile.pascal` to compile
  * type `./ljmd 40 10000` to run
  * type `plot_results` in Matlab command window to check the results
  
  
## 5. Summary of my testing results

### 4.1. Vector addition (chapter 5)

* Array length = 1.0e8.
* CPU (my laptop) function takes 60 ms and 120 ms using single and double precisions, respectively. 
* Computation times using different GPUs are listed in the table below:

|  V100 (S) | V100 (D) | 2080ti (S) | 2080ti (D) | P100 (S) | P100 (D) | laptop-2070 (S) | laptop-2070 (D) | K40 (S) | K40 (D) |
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
| 1.5 ms | 3.0 ms |  2.1 ms |  4.3 ms | 2.2 ms |  4.3 ms | 3.3 ms | 6.8 ms | 6.5 ms | 13 ms |

* If we include cudaMemcpy, GeForce RTX 2080ti takes 130 ms and 250 ms using single and double precisions, respectively. Slower than the CPU!

* If we include cudaMemcpy, GeForce RTX 2070-laptop takes 180 ms and 360 ms using single and double precisions, respectively. Slower than the CPU! Results from using `nvprof`:

`
Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   47.00%  134.38ms         2  67.191ms  62.854ms  71.527ms  [CUDA memcpy HtoD]
                   40.13%  114.74ms         1  114.74ms  114.74ms  114.74ms  [CUDA memcpy DtoH]
                   12.86%  36.778ms        11  3.3435ms  3.3424ms  3.3501ms  add(float const *, float const *, float*, int)
`



### 4.2. A function with high arithmetic intensity (chapter 5)
* CPU function (with an array length of 10^4) takes 320 ms and 450 ms using single and double precisions, respectively. 
* GeForce RTX 2080ti (with an array length of 10^6) takes 15 ms and 450 ms using single and double precisions, respectively.
* Tesla V100 (with an array length of 10^6) takes 11 ms and 28 ms using single and double precisions, respectively.
* GeForce RTX 2070-laptop (with an array length of 10^6) takes 28 ms and 1000 ms using single and double precisions, respectively.
*  GeForce RTX 2070-laptop with single precision:
| N    | time |
|:-------|:-------|
| 1000    | 0.91 ms | 
| 10000   | 0.99 ms | 
| 100000  | 3.8 ms | 
| 1000000       | 28 ms |
| 10000000   | 250 ms | 
| 100000000  | 2500 ms |

### 4.3. Matrix copy and transpose (chapters 7 and 8)

| computation     | V100 (S) | V100 (D) | 2080ti (S) | 2080ti (D) | K40 (S) |
|:---------------------------------|:-------|:-------|:-------|:-------|:-------|
| matrix copy                      | 1.1 ms | 2.0 ms | 1.6 ms | 2.9 ms |  | 
| transpose with coalesced read    | 4.5 ms | 6.2 ms | 5.3 ms | 5.4 ms | 12 ms | 
| transpose with coalesced write   | 1.6 ms | 2.2 ms | 2.8 ms | 3.7 ms | 23 ms | 
| transpose with ldg read          | 1.6 ms | 2.2 ms | 2.8 ms | 3.7 ms | 8 ms |
| transpose with bank conflict     | 1.8 ms | 2.6 ms | 3.5 ms | 4.3 ms |  | 
| transpose without bank conflict  | 1.4 ms | 2.5 ms | 2.3 ms | 4.2 ms |  |


### 4.4. Reduction (chapters 8-10 and 14)

* Array length = 1.0e8 and each element has a value of 1.23.
* The correct summation should be 123000000.
* Using single precision with both CPU and GPU (Tesla K40).

| computation & machine                         | time    |   result  |
|:----------------------------------------------|:--------|:----------|
| CPU with naive summation                      | 85 ms   | 33554432  | 
| global memory only                            | 16.3 ms | 123633392 | 
| static shared memory                          | 10.8 ms | 123633392 | 
| dynamic shared memory                         | 10.8 ms | 123633392 |  
| atomicAdd                                     | 9.8 ms  | 123633392 | 
| atomicAdd and syncwarp                        | 8.1 ms  | 123633392 | 
| atomicAdd and shfl                            | 6.3 ms  | 123633392 | 
| atomicAdd and CP                              | 6.3 ms  | 123633392 | 
| two kernels and less blocks                   | 2.8 ms  | 122999920 | 
| two kernels and less blocks and no cudaMalloc | 2.6 ms  | 122999920 |


### 4.5. Neighbor list construction (chapter 9)

* Number of atoms = 22464.
* CPU function takes 230 ms for both single and double precisions.
* GPU timing results are list in the following table:

| computation     | V100 (S) | V100 (D) | K40 (S) | K40 (D) | 
|:----------------|:---------|:---------|:-----------|:-----------|
| neighbor without atomicAdd | 1.9 ms | 2.6  ms | 10.1 ms | 10.9 ms |
| neighbor with atomicAdd    | 1.8 ms | 2.6  ms | 10.5 ms | 14.5 ms |






