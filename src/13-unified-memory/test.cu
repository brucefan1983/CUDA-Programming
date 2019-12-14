#include "error.cuh"
#include <stdio.h>

int main(void)
{
    long long GB = 1 << 30;
    for (long long n = 1; n <= 1000L; ++n)
    {
        char *d_x;
#ifdef UNIFIED
        CHECK(cudaMallocManaged(&d_x, GB * n));
        printf("Can allocate %lld GB unified memory.\n", n);
#else
        CHECK(cudaMalloc(&d_x, GB * n));
        printf("Can allocate %lld GB device memory.\n", n);
#endif
        CHECK(cudaFree(d_x));
    }
    return 0;
}


