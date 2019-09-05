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
    __shared__ real s_y[128];

    real y = 0.0;
    int offset = tid + bid * blockDim.x * number_of_rounds;
    for (int round = 0; round < number_of_rounds; ++round)
    {
        int n = round * blockDim.x + offset;
        if (n < N) { y += g_x[n]; }
    }
    s_y[tid] = y;
    __syncthreads();

    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_y[tid] += s_y[tid + offset]; }
        __syncthreads();
    }

    if (tid == 0) { g_y[bid] = s_y[0]; }
}

void __global__ reduce_2 
(real *g_x, real *g_sum, int N, int number_of_rounds)
{
    int tid = threadIdx.x;
    __shared__ real s_sum[1024];
    real tmp_sum = 0.0;
    for (int round = 0; round < number_of_rounds; ++round)
    {
        int n = tid + round * 1024;
        if (n < N) { tmp_sum += g_x[n]; }
    }
    s_sum[tid] = tmp_sum;
    __syncthreads();

    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_sum[tid] += s_sum[tid + offset]; }
        __syncthreads();
    }
    
    if (tid == 0) { g_sum[0] = s_sum[0]; }
}

real reduce(real *x, int N, int M)
{
    int block_size = 128;
    int grid_size = (N - 1) / (block_size * M) + 1;
    int number_of_rounds = (grid_size - 1) / 1024 + 1;

    real *y, *sum;
    CHECK(cudaMallocManaged(&y, sizeof(real) * grid_size))
    CHECK(cudaMallocManaged(&sum, sizeof(real)))

    reduce_1<<<grid_size, block_size>>>(x, y, N, M);
    reduce_2<<<1, 1024>>>(y, sum, grid_size, number_of_rounds);

    CHECK(cudaDeviceSynchronize())
    real result = sum[0];
    CHECK(cudaFree(y))
    CHECK(cudaFree(sum))
    return result;
}

