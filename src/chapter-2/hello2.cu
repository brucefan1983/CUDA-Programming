#include <stdio.h>
__global__ void hello_from_gpu(void)
{
    printf("Hello World from the GPU!\n");
}
int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceReset();
    return 0;
}

