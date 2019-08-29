# Chapter 9: using shared memory: reduction

## Source files for the chapter


| file   |      compile      |  run | what to learn |
|----------|:-------------|:----------------|:----------------|
| reduce1.cpp |  g++ -O3 reduce1.cpp | ./a.out | Timing C++ code |
| reduce2one_kernel.cu | nvcc -arch=sm_35 reduce2one_kernel.cu | nvprof --unified-memory-profiling off ./a.out | A slow version for reduction using one kernel |
| reduce3two_kernels.cu | nvcc -arch=sm_35 reduce3two_kernels.cu | nvprof --unified-memory-profiling off ./a.out | A faster version for reduction using two kernels |
| reduce4more_parallelism.cu | nvcc -arch=sm_35 reduce4more_parallelism.cu | nvprof --unified-memory-profiling off ./a.out | An even faster version for reduction with more parallelism |

