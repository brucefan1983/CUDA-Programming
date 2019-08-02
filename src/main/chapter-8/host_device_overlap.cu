#include <math.h> // fabs()
#include <stdio.h>
#include "error.cuh"
#define EPSILON 1.0e-14 // a small number
void cpu_sum(double *x, double *y, double *z, int N);
void __global__ sum(double *x, double *y, double *z, int N);
void run(bool with_cpu);

int main(void)
{
    printf("When the host function is excluded,\n");
    run(false);
    printf("When the host function is included,\n");
    run(true);
    return 0;
}

void cpu_sum(double *x, double *y, double *z, int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void __global__ sum(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}

void run(bool with_cpu)
{
    int N = 100000000;
    int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
        y[n] = 2.0;
        z[n] = 0.0;
    }
    double *g_x, *g_y, *g_z;
    CHECK(cudaMalloc((void **)&g_x, M))
    CHECK(cudaMalloc((void **)&g_y, M))
    CHECK(cudaMalloc((void **)&g_z, M))
    CHECK(cudaMemcpy(g_x, x, M, cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(g_y, y, M, cudaMemcpyHostToDevice))

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;

    cudaDeviceSynchronize();
    clock_t time_begin = clock();
    sum<<<grid_size, block_size>>>(g_x, g_y, g_z, N);

    if (with_cpu)
    {
        cpu_sum(x, y, z, N/25);
    }

    cudaDeviceSynchronize();
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin)
        / double(CLOCKS_PER_SEC);
    printf("time used is = %f s.\n", time_used);

    CHECK(cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost))
    free(x);
    free(y);
    free(z);
    CHECK(cudaFree(g_x))
    CHECK(cudaFree(g_y))
    CHECK(cudaFree(g_z))
}

