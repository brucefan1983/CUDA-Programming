# Chapter 8: using shared memory: matrix transpose


## Source files for the chapter


| file   |      compile      |  run | what to learn |
|----------|:-------------|:-----------|:-----------|
| copy.cu |  nvcc -arch=sm_35 copy.cu | nvprof --unified-memory-profiling off ./a.out | get the effective bandwidth for matrix copying |

