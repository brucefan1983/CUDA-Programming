#include "error.cuh"
#include <stdio.h>
#include <stdint.h>

const int N = 30;

__global__ void gpu_touch(uint64_t *x, const size_t size)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        x[i] = 0;
    }
}

int main(void)
{
    for (int n = 1; n <= N; ++n)
    {
        const size_t memory_size = size_t(n) * 1024 * 1024 * 1024;
        const size_t data_size = memory_size / sizeof(uint64_t);
        uint64_t *x;
        CHECK(cudaMallocManaged(&x, memory_size));
        gpu_touch<<<(data_size - 1) / 1024 + 1, 1024>>>(x, data_size);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(x));
        printf("Allocated %d GB unified memory with GPU touch.\n", n);
    }
    return 0;
}


