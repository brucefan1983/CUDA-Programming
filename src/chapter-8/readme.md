# Chapter 8: Using CUDA libraries

## Source files for the chapter

| file   |      compile      |  run |
|----------|:-------------|------:|
| curand_host1.cu |  nvcc -arch=sm_35 -lcurand curand_host1.cu | ./a.out |
| curand_host2.cu |  nvcc -arch=sm_35 -lcurand curand_host2.cu | ./a.out |
| curand_host3.cu |  nvcc -arch=sm_35 -lcurand curand_host3.cu | ./a.out |
| plot_results.m |  not needed plot_results in Matlab command window |
