# Chapter 4: The crucial ingredients for obtaining speedup

## Source files for the chapter


| file   |      compile      |  run | what to learn |
|----------|:-------------|:----------------|:----------------|
| sum.cpp |  g++ -O3 sum.cpp | ./a.out | Timing C++ code |
| sum.cu |  nvcc -arch=sm_35 sum.cu | nvprof --unified-memory-profiling off ./a.out num_of_repeats| Timing CUDA code using nvprof|
| sum_device_function.cu |  nvcc -arch=sm_35 sum_device_function.cu | nvprof --unified-memory-profiling off ./a.out | How to write a __device__ function?|
| copy.cu |  nvcc -arch=sm_35 copy.cu | nvprof --unified-memory-profiling off ./a.out | Theoretical and effective memory bandwidth|
| pow.cpp |  g++ -O3 pow.cpp | ./a.out | pow() function in C++ |
| pow.cu |  nvcc -arch=sm_35 pow.cu | nvprof --unified-memory-profiling off ./a.out N block_size| pow() function in CUDA |
| plot_results.m |  not needed | type plot_array_size in Matlab command window | How to choose block size? |

