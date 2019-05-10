# Chapter 6: Using shared memory properly

## Source files for the chapter

| file   |      compile      |  run |
|----------|:-------------|------:|
| length1.cpp |  g++ -O3 length1.cpp | ./a.out |
| length2wrong.cu |  nvcc -arch=sm_35 -O3 length2wrong.cu | nvprof --unified-memory-profiling off ./a.out |
| length3.cu |  nvcc -arch=sm_35 -O3 length3.cu | nvprof --unified-memory-profiling off ./a.out |
| length4.cu |  nvcc -arch=sm_35 -O3 length4.cu | nvprof --unified-memory-profiling off ./a.out |
| length5.cu |  nvcc -arch=sm_35 -O3 length5.cu | nvprof --unified-memory-profiling off ./a.out |


