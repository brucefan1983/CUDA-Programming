# Chapter 11: Using warp-level functions

## Source files for this chapter

| file   |      compile      |  run |  what to learn |
|----------|:-------------|:----------------------------|:---------------------------------------------------|
| length7syncwarp.cu |  nvcc -arch=sm_35 length7syncwarp.cu | nvprof --unified-memory-profiling off ./a.out | using the `__syncwarp()` function instead of the `__syncthreads()` function when it is possible|
| length8shfl.cu |  nvcc -arch=sm_35 length8shfl.cu | nvprof --unified-memory-profiling off ./a.out | using the `__shfl_down_sync()` function or the `__shfl_xor_sync()` function |

