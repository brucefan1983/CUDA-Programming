#include <stdio.h>
__global__ void hello_from_gpu(void)
{
    int b = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}
int main(void)
{
    dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceReset();
    return 0;
}

