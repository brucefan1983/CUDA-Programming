# Chapter 4: The crucial ingredients for obtaining speedup

## Source files for the chapter


| file   |      compile      |  run |
|----------|:-------------|------:|
| sum.cpp |  g++ -O3 sum.cpp | ./a.out |
| sum.cu |  nvcc -arch=sm_35 sum.cu | nvprof --unified-memory-profiling off ./a.out num_of_repeats|
| sum_device_function.cu |  nvcc -arch=sm_35 sum_device_function.cu | nvprof --unified-memory-profiling off ./a.out |
| copy.cu |  nvcc -arch=sm_35 copy.cu | nvprof --unified-memory-profiling off ./a.out |
| pow.cpp |  g++ -O3 pow.cpp | ./a.out |
| pow.cu |  nvcc -arch=sm_35 pow.cu | nvprof --unified-memory-profiling off ./a.out N block_size|
| plot_results.m |  not needed | type plot_array_size in Matlab command window |

