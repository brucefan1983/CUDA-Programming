#include "error.cuh"
#include <stdio.h>
double reduce(double *x, int N);

int main(void)
{
    int N = 100000000;
    int M = sizeof(double) * N;
    double *x;
    CHECK(cudaMallocManaged(&x, M))
    for (int n = 0; n < N; ++n) { x[n] = 1.0; }

    double sum = reduce(x, N);
    printf("sum = %g.\n", sum);
    CHECK(cudaFree(x))
    return 0;
}

void __global__ reduce_1(double *g_x, double *g_y, int N)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = bid * blockDim.x + tid;
    __shared__ double s_y[128];
    s_y[tid] = 0.0;
    if (n < N) { s_y[tid] += g_x[n]; }
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
(double *g_x, double *g_sum, int N, int number_of_rounds)
{
    int tid = threadIdx.x;
    __shared__ double s_sum[1024];
    double tmp_sum = 0.0;
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

double reduce(double *x, int N)
{
    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    int number_of_rounds = (grid_size - 1) / 1024 + 1;

    double *y, *sum;
    CHECK(cudaMallocManaged(&y, sizeof(double) * grid_size))
    CHECK(cudaMallocManaged(&sum, sizeof(double)))

    reduce_1<<<grid_size, block_size>>>(x, y, N);
    reduce_2<<<1, 1024>>>(y, sum, grid_size, number_of_rounds);

    CHECK(cudaDeviceSynchronize())
    double result = sum[0];
    CHECK(cudaFree(y))
    CHECK(cudaFree(sum))
    return result;
}

