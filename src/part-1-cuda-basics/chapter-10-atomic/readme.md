# Chapter 10: Using atomics

## Source files for this chapter

| file   |      compile      |  run |  what to learn |
|----------|:-------------|:----------------------------|:---------------------------------------------------|

| length8atomic.cu |  nvcc -arch=sm_35 length8atomic.cu | nvprof --unified-memory-profiling off ./a.out | using the `atomicAdd()` function |

