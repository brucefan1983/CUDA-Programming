# Chapter 3: The basic framework of a CUDA program

## Source files for the chapter


| file   |      compile      |  run |
|----------|:-------------|------:|
| sum.cpp |  g++ -O3 sum.cpp | ./a.out |
| sum1.cu |  nvcc -arch=sm_35 sum1.cu | ./a.out |
| sum2wrong.cu |  nvcc -arch=sm_35 sum2wrong.cu | ./a.out |
| sum3if.cu |  nvcc -arch=sm_35 sum3if.cu | ./a.out |
| sum4check_api.cu |  nvcc -arch=sm_35 sum4check_api.cu | ./a.out |
| sum5check_kernel.cu |  nvcc -arch=sm_35 sum5check_kernel.cu | ./a.out |
