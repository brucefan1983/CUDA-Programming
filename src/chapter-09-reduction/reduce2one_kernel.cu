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

void __global__ reduce
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
    double *sum;
    CHECK(cudaMallocManaged(&sum, sizeof(double)))
    
    int number_of_rounds = (N - 1) / 1024 + 1;
    reduce<<<1, 1024>>>(x, sum, N, number_of_rounds);

    CHECK(cudaDeviceSynchronize())
    double result = sum[0];
    CHECK(cudaFree(sum))
    return result;
}

