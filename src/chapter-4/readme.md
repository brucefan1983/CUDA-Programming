# Chapter 4: The crucial ingredients for obtaining speedup

## Source files for the chapter


| file   |      compile      |  run |
|----------|:-------------|------:|
| sum.cpp |  g++ -O3 sum.cpp | ./a.out |
| sum1.cu |  nvcc -arch=sm_35 sum1.cu | ./a.out |
| sum2.cu |  nvcc -arch=sm_35 sum2.cu | ./a.out |
| pow.cpp |  g++ -O3 pow.cpp | ./a.out |
| pow.cu |  nvcc -arch=sm_35 pow.cu | ./a.out |
| plot_array_size.m |  not needed | plot_array_size |
| plot_block_size.m |  not needed | plot_block_size |
