# Chapter 6: Using global memory and registers properly

## Source files for the chapter

| file   |      compile      |  run |
|----------|:-------------|------:|
| xy.txt |  not applicable | just make sure that it exists |
| neighbor1.cpp |  g++ -O3 neighbor1.cpp | ./a.out |
| neighbor2.cpp |  g++ -O3 neighbor2.cpp | ./a.out |
| neighbor3.cpp |  g++ -O3 neighbor3.cpp | ./a.out |
| neighbor4wrong.cu |  nvcc -arch=sm_35 neighbor4wrong.cu | nvprof --unified-memory-profiling off ./a.out |
| neighbor5.cu |  nvcc -arch=sm_35 neighbor5.cu | nvprof --unified-memory-profiling off ./a.out |
| neighbor6.cu |  nvcc -arch=sm_35 neighbor6.cu | nvprof --unified-memory-profiling off ./a.out |
| neighbor7.cu |  nvcc -arch=sm_35 neighbor7.cu | nvprof --unified-memory-profiling off ./a.out |
| neighbor8.cu |  nvcc -arch=sm_35 neighbor8.cu | nvprof --unified-memory-profiling off ./a.out |
| neighbor9.cu |  nvcc -arch=sm_35 neighbor9.cu | nvprof --unified-memory-profiling off ./a.out |
| neighbor10.cu |  nvcc -arch=sm_35 neighbor10.cu | nvprof --unified-memory-profiling off ./a.out |
| plot_xy_without_bonds.m |  not needed | type plot_xy_without_bonds in Matlab command window |
| plot_xy_with_bonds.m |  not needed | type plot_xy_with_bonds in Matlab command window |

