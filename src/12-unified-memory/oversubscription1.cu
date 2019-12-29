#include "error.cuh"
#include <stdio.h>
#include <stdint.h>

const int N = 30;

int main(void)
{
    for (int n = 1; n <= N; ++n)
    {
        const size_t size = size_t(n) * 1024 * 1024 * 1024;
        uint64_t *x;
#ifdef UNIFIED
        CHECK(cudaMallocManaged(&x, size));
        CHECK(cudaFree(x));
        printf("Allocated %d GB unified memory without touch.\n", n);
#else
        CHECK(cudaMalloc(&x, size));
        CHECK(cudaFree(x));
        printf("Allocate %d GB device memory.\n", n);
#endif
    }
    return 0;
}


