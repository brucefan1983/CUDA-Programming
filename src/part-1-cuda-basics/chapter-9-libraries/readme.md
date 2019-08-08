# Chapter 9: Using CUDA libraries

## Source files for this chapter

| file   |      compile      |  run | description |
|----------|:----------------|-------|------:|
| thrust_scan_vector.cu |  nvcc -arch=sm_35 thrust_scan_vector.cu | ./a.out |
| curand_host1.cu |  nvcc -arch=sm_35 -lcurand curand_host1.cu | ./a.out |
| curand_host2.cu |  nvcc -arch=sm_35 -lcurand curand_host2.cu | ./a.out |
| curand_host3.cu |  nvcc -arch=sm_35 -lcurand curand_host3.cu | ./a.out |
| plot_results.m |  not needed | type plot_results in Matlab command window |
