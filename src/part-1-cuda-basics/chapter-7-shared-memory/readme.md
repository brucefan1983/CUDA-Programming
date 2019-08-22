# Chapter 7: Using shared memory properly

## Source files for this chapter

| file   |      compile      |  run |  what to learn |
|----------|:-------------|:----------------------------|:---------------------------------------------------|
| length1.cpp |  g++ -O3 length1.cpp | ./a.out | C++ base code |
| length2wrong.cu |  nvcc -arch=sm_35 length2wrong.cu | nvprof --unified-memory-profiling off ./a.out | without synchronization |
| length3sync.cu |  nvcc -arch=sm_35 length3sync.cu | nvprof --unified-memory-profiling off ./a.out | add synchronization |
| length4shared.cu |  nvcc -arch=sm_35 length4shared.cu | nvprof --unified-memory-profiling off ./a.out | start to use shared memory |
| length5two_kernels.cu |  nvcc -arch=sm_35 length5two_kernels.cu | nvprof --unified-memory-profiling off ./a.out | use two kernels |
| length6unroll.cu |  nvcc -arch=sm_35 length6unroll.cu | nvprof --unified-memory-profiling off ./a.out | loop unrolling |
| length7template.cu |  nvcc -arch=sm_35 length7template.cu | nvprof --unified-memory-profiling off ./a.out | templating some parameters |
| length8atomic.cu |  nvcc -arch=sm_35 length8atomic.cu | nvprof --unified-memory-profiling off ./a.out | using atomic function |
| length9warp.cu |  nvcc -arch=sm_35 length9warp.cu | nvprof --unified-memory-profiling off ./a.out | using `__warpsync()` |

