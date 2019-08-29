# Chapter 10: Using atomics

## Source files for this chapter


| file   |      compile      |  run | what to learn |
|----------|:-------------|:----------------|:----------------|

| reduce5atomic_global.cu | nvcc -arch=sm_35 reduce5atomic_global.cu | nvprof --unified-memory-profiling off ./a.out | Using `atomicAdd()` with global memory address |
| reduce5atomic_shared.cu | nvcc -arch=sm_35 reduce5atomic_shared.cu | nvprof --unified-memory-profiling off ./a.out | Using `atomicAdd()` with shared memory address |
