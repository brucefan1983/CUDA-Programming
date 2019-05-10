# Chapter 3: The basic framework of a CUDA program

## Source files for the chapter


| file   |      compile      |  run |
|----------|:-------------|------:|
| sum.cpp |  g++ -O3 sum.cpp | ./a.out |
| sum1.cu |  nvcc -arch=sm_35 sum1.cu | ./a.out |
| sum2.cu |  nvcc -arch=sm_35 sum2.cu | ./a.out |
| sum3wrong.cu |  nvcc -arch=sm_35 sum3wrong.cu | ./a.out |
| sum4wrong.cu |  nvcc -arch=sm_35 sum4wrong.cu | ./a.out |
| sum5slow.cu |  nvcc -arch=sm_35 sum5slow.cu | ./a.out |
| sum6.cu |  nvcc -arch=sm_35 sum6.cu | ./a.out |
