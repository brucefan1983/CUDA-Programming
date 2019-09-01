# Chapter 11: Using warp-level functions

## Source files for this chapter

| file   |      compile      |  run |  what to learn |
|----------|:-------------|:----------------------------|:---------------------------------------------------|
| length9warp.cu |  nvcc -arch=sm_35 length9warp.cu | nvprof --unified-memory-profiling off ./a.out | using the `__syncwarp()` function|

