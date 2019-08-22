# Chapter 10: Using CUDA libraries

## Source files for this chapter

| file   |      compile      |  run | what ot learn |
|----------|:----------------|:-------|:-------------------------------------|
| thrust_scan_vector.cu |  nvcc -arch=sm_35 thrust_scan_vector.cu | ./a.out | using the device vector in thrust |
| thrust_scan_pointer.cu |  nvcc -arch=sm_35 thrust_scan_pointer.cu | ./a.out | using the device pointer in thrust |
| cublas_gemm.cu |  nvcc -arch=sm_35 -lcublas cublas_gemm.cu | ./a.out | matrix multiplication in cuBLAS |
| cusolver.cu |  nvcc -arch=sm_35 -lcusolver cusolver.cu | ./a.out | matrix eigenvalues in cuSolver |
| curand_host1.cu |  nvcc -arch=sm_35 -lcurand curand_host1.cu | ./a.out | uniform random numbers in cuRAND |
| curand_host2.cu |  nvcc -arch=sm_35 -lcurand curand_host2.cu | ./a.out | Gaussian random numbers in cuRAND |
| curand_host3.cu |  nvcc -arch=sm_35 -lcurand curand_host3.cu | ./a.out | Gaussian random numbers in cuRAND |
| plot_results.m |  not needed | type plot_results in Matlab command window | check the results form cuRAND |
