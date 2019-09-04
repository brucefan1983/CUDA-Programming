# Chapter 5: The crucial ingredients for obtaining speedup

## Source files for the chapter


| file   |      compile      |  run | what to learn |
|----------|:-------------|:----------------|:----------------|
| sum.cpp |  g++ -O3 sum.cpp | ./a.out | Timing C++ code |
| sum.cu |  nvcc -arch=sm_75 sum.cu | nvprof ./a.out num_of_repeats| Timing CUDA code using nvprof|
| sum_device_function.cu |  nvcc -arch=sm_75 sum_device_function.cu | nvprof ./a.out | How to write a device function?|
| copy.cu |  nvcc -arch=sm_75 copy.cu | nvprof ./a.out | Theoretical and effective memory bandwidths|
| arithmetic.cpp |  g++ -O3 arithmetic.cpp | ./a.out | increasing arithmetic intensity in C++ |
| arithmetic.cu |  nvcc -arch=sm_75 arithmetic.cu | nvprof ./a.out N block_size| increasing arithmetic intensity in CUDA |
| plot_results.m |  not needed | type plot_array_size in Matlab command window | How to choose a good block size? |

