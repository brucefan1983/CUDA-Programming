# Chapter 9: Using CUDA libraries

## Source files for this chapter

| file   |      compile      |  run |
|----------|:----------------|:-------|
| thrust_scan_vector.cu |  nvcc -arch=sm_35 thrust_scan_vector.cu | ./a.out |
| thrust_scan_pointer.cu |  nvcc -arch=sm_35 thrust_scan_pointer.cu | ./a.out |
| cublas_gemm.cu |  nvcc -arch=sm_35 -lcublas cublas_gemm.cu | ./a.out |
| cusolver.cu |  nvcc -arch=sm_35 -lcusolver cusolver.cu | ./a.out |
| curand_host1.cu |  nvcc -arch=sm_35 -lcurand curand_host1.cu | ./a.out |
| curand_host2.cu |  nvcc -arch=sm_35 -lcurand curand_host2.cu | ./a.out |
| curand_host3.cu |  nvcc -arch=sm_35 -lcurand curand_host3.cu | ./a.out |
| plot_results.m |  not needed | type plot_results in Matlab command window |
