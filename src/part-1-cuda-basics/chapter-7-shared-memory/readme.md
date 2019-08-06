# Chapter 7: Using shared memory properly

## Source files for this chapter

| file   |      compile      |  run |
|----------|:-------------|------:|
| length1.cpp |  g++ -O3 length1.cpp | ./a.out |
| length2wrong.cu |  nvcc -arch=sm_35 -O3 length2wrong.cu | nvprof --unified-memory-profiling off ./a.out |
| length3sync.cu |  nvcc -arch=sm_35 -O3 length3sync.cu | nvprof --unified-memory-profiling off ./a.out |
| length4shared.cu |  nvcc -arch=sm_35 -O3 length4shared.cu | nvprof --unified-memory-profiling off ./a.out |
| length5unroll.cu |  nvcc -arch=sm_35 -O3 length5unroll.cu | nvprof --unified-memory-profiling off ./a.out |

