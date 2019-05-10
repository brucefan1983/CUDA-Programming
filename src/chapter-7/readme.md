# Chapter 7: Using shared memory properly

## Source files for the chapter

| file   |      compile      |  run |
|----------|:-------------|------:|
| length1.cpp |  g++ -O3 length1.cpp | ./a.out |
| length2wrong.cu |  nvcc -arch=sm_35 -O3 length2wrong.cu | nvprof --unified-memory-profiling off ./a.out |
| length3.cu |  nvcc -arch=sm_35 -O3 length3.cu | nvprof --unified-memory-profiling off ./a.out |
| length4.cu |  nvcc -arch=sm_35 -O3 length4.cu | nvprof --unified-memory-profiling off ./a.out |
| length5.cu |  nvcc -arch=sm_35 -O3 length5.cu | nvprof --unified-memory-profiling off ./a.out |
| length6.cpp |  g++ -O3 length6.cpp | ./a.out |
| length7.cu |  nvcc -arch=sm_35 -O3 length7.cu | nvprof --unified-memory-profiling off ./a.out |
| length8.cu |  nvcc -arch=sm_35 -O3 length8.cu | nvprof --unified-memory-profiling off ./a.out |
| length9.cu |  nvcc -arch=sm_35 -O3 length9.cu | nvprof --unified-memory-profiling off ./a.out |
| length10.cu |  nvcc -arch=sm_35 -O3 length10.cu | nvprof --unified-memory-profiling off ./a.out |
| length11.cu |  nvcc -arch=sm_35 -O3 length11.cu | nvprof --unified-memory-profiling off ./a.out |
| length12.cu |  nvcc -arch=sm_35 -O3 length12.cu | nvprof --unified-memory-profiling off ./a.out |


