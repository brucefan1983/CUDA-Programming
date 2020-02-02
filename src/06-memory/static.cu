#include "error.cuh"
#include <stdio.h>
__device__ int x = 1;
__device__ int y[2];

void __global__ my_kernel(void)
{
    y[0] = x + 1;
    y[1] = x + 2;
    printf("x = %d, y[0] = %d, y[1] = %d.\n", x, y[0], y[1]);
}

int main(void)
{
    my_kernel<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());
    return 0;
}

