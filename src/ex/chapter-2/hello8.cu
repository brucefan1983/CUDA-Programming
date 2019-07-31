#include <stdio.h>
__global__ void hello_from_gpu(void)
{
    int b = blockIdx.x;
    int t = threadIdx.x + threadIdx.y * blockDim.x;
    printf("Hello World from block-%d and thread-%d!\n", b, t);
}
int main(void)
{
    dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceReset();
    return 0;
}

