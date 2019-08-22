# Chapter 8: CUDA streams 

## Source files for this chapter

| file   |      compile      |  run | what to learn |
|----------|:-------------|:-------------|:---------------------|
| host_kernel.cu |  nvcc -arch=sm_35 host_kernel.cu | ./a.out | overlapping host and device computations |
| kernel_kernel.cu |  nvcc -arch=sm_35 kernel_kernel.cu | ./a.out > t.txt | overlaping multiple kernels |
| kernel_transfer.cu |  nvcc -arch=sm_35 kernel_transfer.cu | ./a.out > t2.txt| overlaping kernel execution and memory transfer |
| plot_t.m |  not needed | type plot_t in the command window of Matlab | what is the optimimal number of streams? | 
| plot_t2.m |  not needed | type plot_t2 in the command window of Matlab | what is the optimimal number of streams? | 
