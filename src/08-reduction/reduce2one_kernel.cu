#include "error.cuh"
#include <stdio.h>
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif
real reduce(real *x, int N);

int main(void)
{
    int N = 100000000;
    int M = sizeof(real) * N;
    real *h_x = (real *)malloc(M);
    for (int n = 0; n < N; ++n) { h_x[n] = 1.0; }
    real *x;
    CHECK(cudaMalloc(&x, M))
    CHECK(cudaMemcpy(x, h_x, M, cudaMemcpyHostToDevice))

    real sum = reduce(x, N);
    printf("sum = %g.\n", sum);

    free(h_x);
    CHECK(cudaFree(x))
    return 0;
}

void __global__ reduce
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

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_sum[tid] += s_sum[tid + offset]; }
        __syncthreads();
    }
    
    if (tid == 0) { g_sum[0] = s_sum[0]; }
}

real reduce(real *x, int N)
{
    real *sum;
    CHECK(cudaMalloc(&sum, sizeof(real)))
    
    int number_of_rounds = (N - 1) / 1024 + 1;
    reduce<<<1, 1024>>>(x, sum, N, number_of_rounds);

    real *h_sum = (real *)malloc(sizeof(real));
    CHECK(cudaMemcpy(h_sum, sum, sizeof(real), 
        cudaMemcpyDeviceToHost))
    real result = h_sum[0];

    free(h_sum);
    CHECK(cudaFree(sum))
    return result;
}

