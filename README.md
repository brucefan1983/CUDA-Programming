# `CUDA`-Programming
Source codes for my `CUDA` programming book

## About the book:
  * To be published by in early 2020.
  * The language for the book is **Chinese**.
  * Covers from Kepler to Volta and is based on `CUDA` 10.1.
  * The book has two parts:
    * Part 1: Teaches `CUDA` programming step by step (14 chapters);
    * part 2: Developing a molecular dynamics code from the ground up (6 chapters) 
  * I assume that the readers
    * have mastered `C` and know some `C++` (for the whole book)
    * have studied **mathematics** at the undergraduate level (for some chapters)
    * have studied **physics** at the undergraduate level (for the second part only)


## Check the following files to learn how to compile and run the source codes
* run_maxwell_chapter7.sh (using my own desktop with Quadro k2000)
* run_kepler.sh (using a workstation with Tesla K40)
* run_pascal.slrum (using Finland's supercomputer with Tesla P100: https://research.csc.fi/taito-gpu)

## Table of contents and List of source codes:


### Chapter 1: Introduction to GPU hardware and `CUDA` programming tools
There is no source code for this chapter.


### Chapter 2: Thread organization in `CUDA`
| file      | what to learn? |
|:----------|:---------------|
| hello.cpp | writing a Hello Word program in `C++` |
| hello1.cu | a valid `C++` program is (usually) also a valid `CUDA` program |
| hello2.cu | write a simple `CUDA` kernel and call `printf()` within it |
| hello3.cu | using multiple threads in a block |
| hello4.cu | using multiple blocks in a grid |
| hello5.cu | using a 2D block |


### Chapter 3: The basic framework of a `CUDA` program
| file         | what to learn ? |
|:-------------|:----------------|
| add.cpp      | adding up two arrays using `C++` |
| add1.cu      | adding up two arrays using `CUDA` |
| add2wrong.cu | what if the memory transfer direction is wrong? |
| add3if.cu    | when do we need an if statement in the kernel? |


### Chapter 4: Error checking
| file                | what to learn ? |
|:--------------------|:----------------|
| add4check_api.cu    | how to check `CUDA` runtime API calls? |
| add5check_kernel.cu | how to check `CUDA` kernel calls? |


### Chapter 5: The crucial ingredients for obtaining speedup
| file                   | what to learn ? |
|:-----------------------|:----------------|
| add.cpp                | timing `C++` code |
| add.cu                 | timing `CUDA` code using nvprof |
| copy.cu                | theoretical and effective memory bandwidths |
| arithmetic.cpp         | increasing arithmetic intensity in `C++` |
| arithmetic.cu          | increasing arithmetic intensity in `CUDA` |


### Chapter 6: Memory organization in `CUDA`
There is no source code for this chapter.


### Chapter 7: Using shared memory: matrix transpose
| file                                 | what to learn? |
|:-------------------------------------|:---------------|
| copy.cu                              | get the effective bandwidth for matrix copying |
| transpose1global_coalesced_read.cu   | coalesced read but non-coalesced write |
| transpose2global_coalesced_write.cu  | coalesced write but non-coalesced read |
| transpose3global_ldg.cu              | using `__ldg` for non-coalesced read (not needed for Pascal) |
| transpose4shared_with_conflict.cu    | using shared memory but with bank conflict |
| transpose5shared_without_conflict.cu | using shared memory and without bank conflict |


### Chapter 8: Using shared memory: reduction
| file                       | what to learn ? |
|:---------------------------| :---------------|
| reduce1.cpp                | reduction in `C++` |
| reduce2one_kernel.cu       | a slow version for reduction using one kernel |
| reduce3two_kernels.cu      | a faster version for reduction using two kernels |
| reduce4more_parallelism.cu | an even faster version for reduction with more parallelism |


### Chapter 9: Using atomic functions
| file             | what to learn ? |
|:-----------------| :---------------|
| reduce5atomic.cu | using `atomicAdd()` to eliminate the second kernel |


### Chapter 10: Using warp-level functions
| file               | what to learn ? |
|:-------------------| :---------------|
| reduce7syncwarp.cu | using the `__syncwarp()` function instead of the `__syncthreads()` function within warps |
| reduce8shfl.cu     | using the `__shfl_down_sync()` or the `__shfl_xor_sync()` function for warp reduction |
| reduce9cp.cu       | using the cooperative groups |


### Chapter 11: `CUDA` streams 
| file               | what to learn ? |
|:-------------------| :---------------|
| host_kernel.cu     | overlapping host and device computations |
| kernel_kernel.cu   | overlaping multiple kernels |
| kernel_transfer.cu | overlaping kernel execution and memory transfer |


### Chapter 12: Using `CUDA` libraries
| file                   | what to learn ? |
|:-----------------------| :---------------|
| thrust_scan_vector.cu  | using the device vector in `thrust` |
| thrust_scan_pointer.cu | using the device pointer in `thrust` |
| cublas_gemm.cu         | matrix multiplication in `cuBLAS` |
| cusolver.cu            | matrix eigenvalues in `cuSolver` |
| curand_host1.cu        | uniform random numbers in `cuRAND` |
| curand_host2.cu        | Gaussian random numbers in `cuRAND` |


### Chapter 13: Unified memory programming 
| file           | what to learn ? |
|:---------------| :---------------|
| add_unified.cu | using unified memory |


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


