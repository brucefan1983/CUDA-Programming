# CUDA-Programming
Source codes for my CUDA programming book

## About the book
  * To be published by in 2020.
  * The language for the book is Chinese.
  * Covers from Kepler to Volta.
  * Based on CUDA 10.1.
  * Teaches CUDA programming step by step.
  * Has a real world project: developing a molecular dynamics code from the ground up.
  * Provides ALL the source codes.
  * The book has two parts:
    * part 1: Basics (14 chapters)
    * part 2: an MD project (6 chapters)
  * The book will have about 300 pages, 200 pages for the first part and 100 pages for the second.
  * I assume that the reders
    * have mastered C and know some C++ (for the whole book)
    * have studied mathematics at the undergraduate level (for some chapters)
    * have studied physics at the undergraduate level (for the second part of the book only)


## List of source codes:


### Chapter 1: Introduction to GPU hardware and CUDA programming tools

There is no source code for this chapter.


### Chapter 2: Thread organization in CUDA

| file      | what to learn? |
|-----------|:---------------|
| hello.cpp | writing a Hello Word program in C++ |
| hello1.cu | a valid C++ program is (usually) also a valid CUDA program |
| hello2.cu | write a simple CUDA kernel and call `printf()` within it |
| hello3.cu | using multiple threads in a block |
| hello4.cu | using multiple blocks in a grid |
| hello5.cu | using a 2D block |


### Chapter 3: The basic framework of a CUDA program

| file         | what to learn ? |
|--------------|:----------------|
| sum.cpp      | adding up two arrays using C++ |
| sum1.cu      | adding up two arrays using CUDA |
| sum2wrong.cu | what if the memory transfer direction is wrong? |
| sum3if.cu    | when do we need an if statement in the kernel? |


### Chapter 4: Error checking

| file                | what to learn ? |
|---------------------|:----------------|
| sum4check_api.cu    | how to check CUDA runtime API calls? |
| sum5check_kernel.cu | how to check CUDA kernel calls? |


### Chapter 5: The crucial ingredients for obtaining speedup

| file                   | what to learn ? |
|------------------------|:----------------|
| add.cpp                | timing C++ code |
| add.cu                 | timing CUDA code using nvprof |
| copy.cu                | theoretical and effective memory bandwidths |
| arithmetic.cpp         | increasing arithmetic intensity in C++ |
| arithmetic.cu          | increasing arithmetic intensity in CUDA |


### Chapter 6: Memory organization in CUDA


### Chapter 7: using shared memory: matrix transpose

| file                                 | what to learn? |
|--------------------------------------|:---------------|
| copy.cu                              | get the effective bandwidth for matrix copying |
| transpose1global_coalesced_read.cu   | coalesced read but non-coalesced write |
| transpose2global_coalesced_write.cu  | coalesced write but non-coalesced read |
| transpose3global_ldg.cu              | using `__ldg` for non-coalesced read (not needed for Pascal) |
| transpose4shared_with_conflict.cu    | using shared memory but with bank conflict |
| transpose5shared_without_conflict.cu | using shared memory and without bank conflict |



