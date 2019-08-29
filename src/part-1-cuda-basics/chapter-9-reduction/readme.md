# Chapter 9: using shared memory: reduction

## Source files for the chapter


| file   |      compile      |  run | what to learn |
|----------|:-------------|:----------------|:----------------|
| reduce1.cpp |  g++ -O3 reduce1.cpp | ./a.out | Timing C++ code |
| reduce2one_kernel.cu |  g++ -O3 reduce2one_kernel.cu | nvprof --unified-memory-profiling off ./a.out | A slow verion for reduction using one kernel |
| reduce3two_kernels.cu |  g++ -O3 reduce3two_kernels.cu | nvprof --unified-memory-profiling off ./a.out | A faster verion for reduction using two kernels |
| reduce4more_parallelism.cu |  g++ -O3 reduce4more_parallelism.cu | nvprof --unified-memory-profiling off ./a.out | An even faster verion for reduction with more parallelism |

