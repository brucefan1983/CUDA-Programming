#include "error.cuh"
#include <stdio.h>
#include <stdint.h>

const int N = 30;

void cpu_touch(uint64_t *x, size_t size)
{
    for (size_t i = 0; i < size / sizeof(uint64_t); i++) 
    {
        x[i] = 0;
    }
}

int main(void)
{
    for (int n = 1; n <= N; ++n)
    {
        size_t size = size_t(n) * 1024 * 1024 * 1024;
        uint64_t *x;
        CHECK(cudaMallocManaged(&x, size));
        cpu_touch(x, size);
        CHECK(cudaFree(x));
        printf("Allocated %d GB unified memory with CPU touch.\n", n);
    }
    return 0;
}


