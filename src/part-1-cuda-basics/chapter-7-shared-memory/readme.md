# Chapter 7: Using shared memory properly

## Source files for this chapter

| file   |      compile      |  run |  what to learn |
|----------|:-------------|:----------------------------|:---------------------------------------------------|
| length1.cpp |  g++ -O3 length1.cpp | ./a.out | C++ base code |
| length2wrong.cu |  nvcc -arch=sm_35 length2wrong.cu | nvprof --unified-memory-profiling off ./a.out | getting wrong results without synchronization |
| length3sync.cu |  nvcc -arch=sm_35 length3sync.cu | nvprof --unified-memory-profiling off ./a.out | add synchronization to make it correct|
| length4shared.cu |  nvcc -arch=sm_35 length4shared.cu | nvprof --unified-memory-profiling off ./a.out | using shared memory to reduce global memory access|
| length5two_kernels.cu |  nvcc -arch=sm_35 length5two_kernels.cu | nvprof --unified-memory-profiling off ./a.out | using two kernels  to increase the parallelism |
| length6unroll.cu |  nvcc -arch=sm_35 length6unroll.cu | nvprof --unified-memory-profiling off ./a.out | loop unrolling for increased throughput |
| length7template.cu |  nvcc -arch=sm_35 length7template.cu | nvprof --unified-memory-profiling off ./a.out | templating some parameters to make the code more readable |
| length8atomic.cu |  nvcc -arch=sm_35 length8atomic.cu | nvprof --unified-memory-profiling off ./a.out | using the `atomicAdd()` function |
| length9warp.cu |  nvcc -arch=sm_35 length9warp.cu | nvprof --unified-memory-profiling off ./a.out | using the `__syncwarp()` function|

