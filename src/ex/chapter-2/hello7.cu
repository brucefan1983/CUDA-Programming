#include <stdio.h>
__global__ void hello_from_gpu(void)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (threadIdx.x % 2 != 0) return; // skip odd threadIdx.x
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}
int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceReset();
    return 0;
}

