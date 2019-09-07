# Chapter 10: Using warp-level functions

## Source files for this chapter

| file               | what to learn ? |
|--------------------| :---------------|
| reduce7syncwarp.cu | using the `__syncwarp()` function instead of the `__syncthreads()` function within warps |
| reduce8shfl.cu     | using the `__shfl_down_sync()` or the `__shfl_xor_sync()` function for warp reduction |
| reduce9cp.cu       | using the cooperative groups |

