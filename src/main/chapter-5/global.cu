#include "error.cuh"

void allocate_memory(int N)
{
    long long GB = (1 << 30); // 2^30 = 1024^3
    long long M = GB * N;
    char *g_x;
    printf("Try to allocate %d GB global memory\n", N);
    CHECK(cudaMalloc((void **)&g_x, M))
    printf("Allocated %d GB global memory\n", N);
    CHECK(cudaFree(g_x))
}

int main(void)
{
    allocate_memory(10); // 10 GB
    allocate_memory(12); // 12 GB
    return 0;
}

