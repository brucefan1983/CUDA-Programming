#include <stdio.h>
double reduce(double *x, int N, int M);

int main(int argc, char **argv)
{
    int M = atoi(argv[1]);
    int N = 100000000;
    double *x;
    cudaMallocManaged(&x, sizeof(double) * N);
    for (int n = 0; n < N; ++n) { x[n] = 1.0; }

    double sum = reduce(x, N, M);
    printf("sum = %g.\n", sum);
    cudaFree(x);
    return 0;
}

void __global__ reduce_1
(double *g_x, double *g_y, int N, int number_of_rounds)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ double s_y[128];

    double y = 0.0;
    int offset = tid + bid * blockDim.x * number_of_rounds;
    for (int round = 0; round < number_of_rounds; ++round)
    {
        int n = round * blockDim.x + offset;
        if (n < N) { y += g_x[n]; }
    }
    s_y[tid] = y;
    __syncthreads();

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

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_sum[tid] += s_sum[tid + offset]; }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        g_sum[0] = s_sum[0];
    }
}

double reduce(double *x, int N, int M)
{
    int block_size = 128;
    int grid_size = (N - 1) / (block_size * M) + 1;
    int number_of_rounds = (grid_size - 1) / 1024 + 1;

    double *y, *sum;
    cudaMallocManaged(&y, sizeof(double) * grid_size);
    cudaMallocManaged(&sum, sizeof(double));

    reduce_1<<<grid_size, block_size>>>(x, y, N, M);
    reduce_2<<<1, 1024>>>(y, sum, grid_size, number_of_rounds);

    cudaDeviceSynchronize();
    double result = sum[0];
    cudaFree(y);
    cudaFree(sum);
    return result;
}

