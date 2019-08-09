# Chapter 7: Using shared memory properly

## Source files for this chapter

| file   |      compile      |  run |
|----------|:-------------|------:|
| length1.cpp |  g++ -O3 length1.cpp | ./a.out |
| length2wrong.cu |  nvcc -arch=sm_35 length2wrong.cu | nvprof --unified-memory-profiling off ./a.out |
| length3sync.cu |  nvcc -arch=sm_35 length3sync.cu | nvprof --unified-memory-profiling off ./a.out |
| length4shared.cu |  nvcc -arch=sm_35 length4shared.cu | nvprof --unified-memory-profiling off ./a.out |
| length5two_kernels.cu |  nvcc -arch=sm_35 length5two_kernels.cu | nvprof --unified-memory-profiling off ./a.out |
| length6unroll.cu |  nvcc -arch=sm_35 length6unroll.cu | nvprof --unified-memory-profiling off ./a.out |
| length7template.cu |  nvcc -arch=sm_35 length7template.cu | nvprof --unified-memory-profiling off ./a.out |
| length8warp.cu |  nvcc -arch=sm_35 length8warp.cu | nvprof --unified-memory-profiling off ./a.out |
| length9divergent.cu |  nvcc -arch=sm_35 length9divergent.cu | nvprof --unified-memory-profiling off ./a.out |

