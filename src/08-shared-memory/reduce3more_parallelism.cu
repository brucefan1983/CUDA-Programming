#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_ROUNDS = 10;

real reduce(real *x, int N);

int main(int argc, char **argv)
{
    int N = 100000000;
    int M = sizeof(real) * N;
    real *h_x = (real *)malloc(M);
    for (int n = 0; n < N; ++n) { h_x[n] = 1.0; }
    real *x;
    CHECK(cudaMalloc(&x, M))
    CHECK(cudaMemcpy(x, h_x, M, cudaMemcpyHostToDevice))

    real sum = reduce(x, N);
    printf("sum = %f.\n", sum);

    free(h_x);
    CHECK(cudaFree(x))
    return 0;
}

void __global__ reduce(real *g_x, real *g_y, int N)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ real s_y[128];

    real y = 0.0;
    int offset = tid + bid * blockDim.x * NUM_ROUNDS;
    for (int round = 0; round < NUM_ROUNDS; ++round)
    {
        int n = round * blockDim.x + offset;
        if (n < N) { y += g_x[n]; }
    }
    s_y[tid] = y;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        __syncthreads();
        if (tid < offset) { s_y[tid] += s_y[tid + offset]; }
    }

    if (tid == 0) { g_y[bid] = s_y[0]; }
}

real reduce(real *x, int N)
{
    const int block_size = 128;
    int grid_size = (N - 1) / (block_size * NUM_ROUNDS) + 1;

    real *y;
    CHECK(cudaMalloc(&y, sizeof(real) * grid_size))

    reduce<<<grid_size, block_size>>>(x, y, N);

    real *h_y = (real *)malloc(sizeof(real) * grid_size);
    CHECK(cudaMemcpy(h_y, y, sizeof(real) * grid_size, 
        cudaMemcpyDeviceToHost))

    real result = 0.0;
    for (int n = 0; n < grid_size; ++n) { result += h_y[n]; }

    free(h_y);
    CHECK(cudaFree(y))
    return result;
}

