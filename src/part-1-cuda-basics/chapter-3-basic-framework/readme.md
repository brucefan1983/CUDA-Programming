# Chapter 3: The basic framework of a CUDA program

## Source files for the chapter


| file   |      compile      |  run | what to learn
|----------|:-------------|------:|:-------------------------|
| sum.cpp |  g++ -O3 sum.cpp | ./a.out | Sum two arrays using C++.|
| sum1.cu |  nvcc -arch=sm_35 sum1.cu | ./a.out | Sum two arrays using CUDA.|
| sum2wrong.cu |  nvcc -arch=sm_35 sum2wrong.cu | ./a.out | What if when the memory transfer direction is wrong? |
| sum3if.cu |  nvcc -arch=sm_35 sum3if.cu | ./a.out | When do we need an if statement in the kernel? |
| sum4check_api.cu |  nvcc -arch=sm_35 sum4check_api.cu | ./a.out | How to check CUDA runtime API calls? |
| sum5check_kernel.cu |  nvcc -arch=sm_35 sum5check_kernel.cu | ./a.out | How to check CUDA kernel calls? |
