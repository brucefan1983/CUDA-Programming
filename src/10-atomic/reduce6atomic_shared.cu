#include "error.cuh"
#include <stdio.h>
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

real reduce(real *x, int N, int M);

int main(int argc, char **argv)
{
    int M = atoi(argv[1]);
    int N = 100000000;
    real *x;
    CHECK(cudaMallocManaged(&x, sizeof(real) * N))
    for (int n = 0; n < N; ++n) { x[n] = 1.0; }

    real sum = reduce(x, N, M);
    printf("sum = %g.\n", sum);
    CHECK(cudaFree(x))
    return 0;
}

void __global__ reduce_1
(real *g_x, real *g_y, int N, int number_of_rounds)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ real s_sum;
    s_sum = 0.0;
    real y = 0.0;
    int offset = tid + bid * blockDim.x * number_of_rounds;
    for (int round = 0; round < number_of_rounds; ++round)
    {
        int n = round * blockDim.x + offset;
        if (n < N) { y += g_x[n]; }
    }
    __syncthreads();

    atomicAdd(&s_sum, y);
    __syncthreads();
    if (tid == 0) { atomicAdd(g_y, s_sum); }
}

real reduce(real *x, int N, int M)
{
    int block_size = 128;
    int grid_size = (N - 1) / (block_size * M) + 1;

    real *sum;
    CHECK(cudaMallocManaged(&sum, sizeof(real)))
    CHECK(cudaDeviceSynchronize())
    sum[0] = 0.0;

    reduce_1<<<grid_size, block_size>>>(x, sum, N, M);

    CHECK(cudaDeviceSynchronize())
    real result = sum[0];
    CHECK(cudaFree(sum))
    return result;
}

